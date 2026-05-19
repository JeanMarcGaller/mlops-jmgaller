import pandas as pd
import pytest

import feature_pipeline
from config import TARGET_COLUMN


def make_raw_weather_df(periods: int = 40) -> pd.DataFrame:
    event_time = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=periods,
        freq="h",
    )

    return pd.DataFrame(
        {
            "location_id": ["basel"] * periods,
            "event_time_unix": (event_time.view("int64") // 10**9).astype(int),
            "event_time": event_time,
            "temperature_2m": [float(i) for i in range(periods)],
            "relative_humidity_2m": [70 + (i % 10) for i in range(periods)],
            "precipitation": [0.1 * (i % 3) for i in range(periods)],
            "cloud_cover": [50 + (i % 20) for i in range(periods)],
            "pressure_msl": [1010.0 + 0.1 * i for i in range(periods)],
            "wind_speed_10m": [5.0 + 0.1 * i for i in range(periods)],
        }
    )


def test_build_and_cache_training_features_creates_features_and_cache(
    monkeypatch,
    tmp_path,
):
    raw_df = make_raw_weather_df(periods=60)
    cache_path = tmp_path / "temperature_features.parquet"

    monkeypatch.setattr(
        feature_pipeline,
        "fetch_historical_weather",
        lambda start_date, end_date: raw_df,
    )
    monkeypatch.setattr(feature_pipeline, "FEATURE_CACHE_PATH", cache_path)
    monkeypatch.setattr(feature_pipeline, "DATA_FEATURES_DIR", tmp_path)

    result = feature_pipeline.build_and_cache_training_features(days=30)

    assert not result.empty
    assert TARGET_COLUMN in result.columns
    assert cache_path.exists()

    cached_df = pd.read_parquet(cache_path)
    assert cached_df.shape == result.shape
    assert cached_df.columns.tolist() == result.columns.tolist()


def test_build_and_cache_training_features_rejects_invalid_days():
    with pytest.raises(ValueError, match="days must be greater than 0"):
        feature_pipeline.build_and_cache_training_features(days=0)


def test_main_dry_run_does_not_write_to_hopsworks(monkeypatch):
    dummy_features = pd.DataFrame(
        {
            "location_id": ["basel"],
            "event_time_unix": [1],
            "event_time": [pd.Timestamp("2026-01-01")],
            TARGET_COLUMN: [10.0],
        }
    )

    called = {"write": False}

    def fake_write_to_feature_store(features_df):
        called["write"] = True

    monkeypatch.setattr(
        feature_pipeline,
        "build_and_cache_training_features",
        lambda days: dummy_features,
    )
    monkeypatch.setattr(
        feature_pipeline,
        "write_to_feature_store",
        fake_write_to_feature_store,
    )

    feature_pipeline.main(days=30, dry_run=True)

    assert called["write"] is False


def test_main_writes_to_hopsworks_when_not_dry_run(monkeypatch):
    dummy_features = pd.DataFrame(
        {
            "location_id": ["basel"],
            "event_time_unix": [1],
            "event_time": [pd.Timestamp("2026-01-01")],
            TARGET_COLUMN: [10.0],
        }
    )

    called = {"write": False}

    def fake_write_to_feature_store(features_df):
        called["write"] = True
        assert features_df.equals(dummy_features)

    monkeypatch.setattr(
        feature_pipeline,
        "build_and_cache_training_features",
        lambda days: dummy_features,
    )
    monkeypatch.setattr(
        feature_pipeline,
        "write_to_feature_store",
        fake_write_to_feature_store,
    )

    feature_pipeline.main(days=30, dry_run=False)

    assert called["write"] is True
