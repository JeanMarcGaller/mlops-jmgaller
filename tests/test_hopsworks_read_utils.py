from pathlib import Path

import pandas as pd
import pytest

import hopsworks_read_utils as utils


class FakeFeatureGroup:
    def __init__(self, responses):
        self.responses = list(responses)
        self.read_calls = 0
        self.received_read_options = []

    def read(self, read_options=None):
        self.read_calls += 1
        self.received_read_options.append(read_options)

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


def validate_identity(df: pd.DataFrame) -> pd.DataFrame:
    return df


def validate_add_marker(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["validated"] = True
    return result


def test_read_feature_group_with_retry_reads_successfully(tmp_path):
    df = pd.DataFrame({"value": [1, 2, 3]})
    feature_group = FakeFeatureGroup([df])

    result = utils.read_feature_group_with_retry(
        feature_group=feature_group,
        cache_path=tmp_path / "cache.parquet",
        validate_fn=validate_identity,
        context="unit test",
        wait_seconds=0,
    )

    assert result.equals(df)
    assert feature_group.read_calls == 1
    assert feature_group.received_read_options[0]["use_hive"] is True
    assert (
        feature_group.received_read_options[0]["arrow_flight_config"]["timeout"]
        == utils.DEFAULT_ARROW_FLIGHT_TIMEOUT_SECONDS
    )


def test_read_feature_group_with_retry_retries_after_failure(tmp_path):
    df = pd.DataFrame({"value": [1, 2, 3]})
    feature_group = FakeFeatureGroup(
        [
            RuntimeError("temporary failure"),
            df,
        ]
    )

    result = utils.read_feature_group_with_retry(
        feature_group=feature_group,
        cache_path=tmp_path / "cache.parquet",
        validate_fn=validate_identity,
        context="unit test",
        wait_seconds=0,
    )

    assert result.equals(df)
    assert feature_group.read_calls == 2


def test_read_feature_group_with_retry_uses_cache_after_failures(tmp_path):
    cache_path = tmp_path / "cache.parquet"
    cached_df = pd.DataFrame({"value": [10, 20, 30]})
    cached_df.to_parquet(cache_path)

    feature_group = FakeFeatureGroup(
        [
            RuntimeError("failure 1"),
            RuntimeError("failure 2"),
            RuntimeError("failure 3"),
        ]
    )

    result = utils.read_feature_group_with_retry(
        feature_group=feature_group,
        cache_path=cache_path,
        validate_fn=validate_add_marker,
        context="unit test",
        wait_seconds=0,
    )

    assert feature_group.read_calls == 3
    assert result["value"].tolist() == [10, 20, 30]
    assert result["validated"].tolist() == [True, True, True]


def test_read_feature_group_with_retry_raises_if_cache_missing(tmp_path):
    feature_group = FakeFeatureGroup(
        [
            RuntimeError("failure 1"),
            RuntimeError("failure 2"),
            RuntimeError("failure 3"),
        ]
    )

    with pytest.raises(FileNotFoundError, match="Local feature cache not found"):
        utils.read_feature_group_with_retry(
            feature_group=feature_group,
            cache_path=tmp_path / "missing_cache.parquet",
            validate_fn=validate_identity,
            context="unit test",
            wait_seconds=0,
        )


def test_read_feature_group_with_retry_rejects_invalid_max_attempts(tmp_path):
    feature_group = FakeFeatureGroup([])

    with pytest.raises(ValueError, match="max_attempts"):
        utils.read_feature_group_with_retry(
            feature_group=feature_group,
            cache_path=tmp_path / "cache.parquet",
            validate_fn=validate_identity,
            context="unit test",
            max_attempts=0,
            wait_seconds=0,
        )


def test_read_feature_group_with_retry_rejects_invalid_wait_seconds(tmp_path):
    feature_group = FakeFeatureGroup([])

    with pytest.raises(ValueError, match="wait_seconds"):
        utils.read_feature_group_with_retry(
            feature_group=feature_group,
            cache_path=tmp_path / "cache.parquet",
            validate_fn=validate_identity,
            context="unit test",
            wait_seconds=-1,
        )


def test_read_dataframe_from_cache_reads_parquet(tmp_path):
    cache_path = tmp_path / "cache.parquet"
    df = pd.DataFrame({"value": [1, 2, 3]})
    df.to_parquet(cache_path)

    result = utils.read_dataframe_from_cache(cache_path)

    assert result.equals(df)


def test_read_dataframe_from_cache_rejects_missing_cache(tmp_path):
    cache_path = Path(tmp_path / "missing.parquet")

    with pytest.raises(FileNotFoundError, match="Local feature cache not found"):
        utils.read_dataframe_from_cache(cache_path)
