import json

import pandas as pd
import pytest

import feature_selection_pipeline as pipeline
from config import TARGET_COLUMN
from features import FEATURE_COLUMNS


def make_training_df(periods: int = 100) -> pd.DataFrame:
    event_time = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=periods,
        freq="h",
    )

    data = {
        "location_id": ["basel"] * periods,
        "event_time_unix": (event_time.view("int64") // 10**9).astype(int),
        "event_time": event_time,
        TARGET_COLUMN: [float(i % 24) for i in range(periods)],
    }

    for index, feature in enumerate(FEATURE_COLUMNS):
        data[feature] = [float((i + index) % 50) for i in range(periods)]

    return pd.DataFrame(data)


def make_importance_df_with_random_probes() -> pd.DataFrame:
    real_features = FEATURE_COLUMNS[:10]

    return pd.DataFrame(
        {
            "feature": [
                *real_features,
                *pipeline.RANDOM_FEATURE_COLUMNS,
            ],
            "importance_mean": [
                0.50,
                0.40,
                0.30,
                0.20,
                0.15,
                0.12,
                0.09,
                0.08,
                0.01,
                -0.01,
                0.10,
                0.07,
                0.05,
            ],
            "importance_std": [0.01] * (len(real_features) + 3),
        }
    )


def test_add_fixed_model_params_adds_reproducible_defaults():
    params = {"max_iter": 100}

    result = pipeline.add_fixed_model_params(params)

    assert result["max_iter"] == 100
    assert result["loss"] == "squared_error"
    assert result["early_stopping"] is False
    assert result["random_state"] == 42


def test_add_random_probe_features_adds_expected_columns():
    df = make_training_df(periods=30)

    result = pipeline.add_random_probe_features(df, random_state=42)

    for column in pipeline.RANDOM_FEATURE_COLUMNS:
        assert column in result.columns

    assert len(result) == len(df)
    assert result[pipeline.RANDOM_FEATURE_COLUMNS].isna().sum().sum() == 0


def test_validate_training_dataframe_sorts_by_event_time():
    df = make_training_df(periods=30)
    shuffled_df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    result = pipeline.validate_training_dataframe(shuffled_df)

    assert result["event_time"].is_monotonic_increasing


def test_validate_training_dataframe_rejects_missing_target():
    df = make_training_df(periods=30).drop(columns=[TARGET_COLUMN])

    with pytest.raises(ValueError, match="Missing target column"):
        pipeline.validate_training_dataframe(df)


def test_validate_training_dataframe_rejects_missing_feature():
    missing_feature = FEATURE_COLUMNS[0]
    df = make_training_df(periods=30).drop(columns=[missing_feature])

    with pytest.raises(ValueError, match="Missing feature columns"):
        pipeline.validate_training_dataframe(df)


def test_validate_training_dataframe_rejects_missing_event_time():
    df = make_training_df(periods=30).drop(columns=["event_time"])

    with pytest.raises(ValueError, match="Missing event_time"):
        pipeline.validate_training_dataframe(df)


def test_time_based_train_valid_test_split_uses_chronological_order():
    df = make_training_df(periods=100)

    train_df, valid_df, test_df = pipeline.time_based_train_valid_test_split(
        df,
        train_fraction=0.70,
        valid_fraction=0.15,
    )

    assert len(train_df) == 70
    assert len(valid_df) == 15
    assert len(test_df) == 15
    assert train_df["event_time"].max() < valid_df["event_time"].min()
    assert valid_df["event_time"].max() < test_df["event_time"].min()


def test_split_features_and_target_uses_given_feature_columns():
    df = make_training_df(periods=30)
    selected_columns = FEATURE_COLUMNS[:5]

    X, y = pipeline.split_features_and_target(df, selected_columns)

    assert X.columns.tolist() == selected_columns
    assert len(X) == 30
    assert len(y) == 30


def test_compute_correlation_reports_returns_expected_outputs():
    df = make_training_df(periods=50)
    feature_columns = FEATURE_COLUMNS[:5]

    correlation_matrix, high_correlation_df, target_correlation_df = (
        pipeline.compute_correlation_reports(
            train_df=df,
            feature_columns=feature_columns,
            threshold=0.90,
        )
    )

    assert correlation_matrix.shape == (len(feature_columns), len(feature_columns))
    assert isinstance(high_correlation_df, pd.DataFrame)
    assert set(target_correlation_df.columns) == {
        "feature",
        "target_correlation",
        "abs_target_correlation",
    }


def test_load_best_params_reads_valid_json(monkeypatch, tmp_path):
    best_params_path = tmp_path / "best_params.json"
    best_params = {
        "max_iter": 100,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
    }

    best_params_path.write_text(
        json.dumps({"best_params": best_params}),
        encoding="utf-8",
    )

    monkeypatch.setattr(pipeline, "BEST_PARAMS_PATH", best_params_path)

    result = pipeline.load_best_params()

    assert result == best_params


def test_load_best_params_rejects_missing_file(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing_best_params.json"
    monkeypatch.setattr(pipeline, "BEST_PARAMS_PATH", missing_path)

    with pytest.raises(FileNotFoundError, match="Best params file not found"):
        pipeline.load_best_params()


def test_load_best_params_rejects_missing_best_params_key(monkeypatch, tmp_path):
    best_params_path = tmp_path / "best_params.json"
    best_params_path.write_text(json.dumps({"other_key": {}}), encoding="utf-8")

    monkeypatch.setattr(pipeline, "BEST_PARAMS_PATH", best_params_path)

    with pytest.raises(ValueError, match="No 'best_params'"):
        pipeline.load_best_params()


def test_get_random_probe_threshold_returns_max_probe_importance():
    importance_df = make_importance_df_with_random_probes()

    threshold = pipeline.get_random_probe_threshold(importance_df)

    assert threshold == pytest.approx(0.10)


def test_get_random_probe_threshold_rejects_missing_random_probes():
    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS[:3],
            "importance_mean": [0.3, 0.2, 0.1],
            "importance_std": [0.01, 0.01, 0.01],
        }
    )

    with pytest.raises(ValueError, match="random probe features"):
        pipeline.get_random_probe_threshold(importance_df)


def test_select_relevant_features_uses_random_probe_threshold():
    importance_df = make_importance_df_with_random_probes()

    selected = pipeline.select_relevant_features(importance_df)

    assert selected == FEATURE_COLUMNS[:8]
    assert not any(feature in selected for feature in pipeline.RANDOM_FEATURE_COLUMNS)


def test_select_relevant_features_uses_minimum_feature_count():
    importance_df = pd.DataFrame(
        {
            "feature": [
                *FEATURE_COLUMNS[:10],
                *pipeline.RANDOM_FEATURE_COLUMNS,
            ],
            "importance_mean": [
                0.50,
                0.01,
                0.00,
                -0.01,
                -0.02,
                -0.03,
                -0.04,
                -0.05,
                -0.06,
                -0.07,
                0.40,
                0.30,
                0.20,
            ],
            "importance_std": [0.01] * 13,
        }
    )

    selected = pipeline.select_relevant_features(importance_df)

    assert selected == FEATURE_COLUMNS[:8]
    assert not any(feature in selected for feature in pipeline.RANDOM_FEATURE_COLUMNS)


def test_build_selected_features_payload_contains_metadata():
    importance_df = make_importance_df_with_random_probes()
    selected_features = FEATURE_COLUMNS[:8]
    metrics = {"valid_mae_before_selection": 1.23}
    best_params = {"max_iter": 100}

    payload = pipeline.build_selected_features_payload(
        importance_df=importance_df,
        selected_features=selected_features,
        metrics=metrics,
        best_params=best_params,
    )

    assert payload["model_type"] == "HistGradientBoostingRegressor"
    assert payload["target_column"] == TARGET_COLUMN
    assert payload["selected_features"] == selected_features
    assert payload["n_selected_features"] == 8
    assert payload["selection_method"] == "permutation_importance_with_random_probes"
    assert payload["random_probe_features"] == pipeline.RANDOM_FEATURE_COLUMNS
    assert payload["random_probe_threshold"] == pytest.approx(0.10)
    assert len(payload["random_probe_importance"]) == 3
    assert payload["correlation_threshold"] == pipeline.CORRELATION_THRESHOLD
    assert payload["best_params"] == best_params
    assert payload["model_params"]["loss"] == "squared_error"
    assert payload["model_params"]["early_stopping"] is False
    assert payload["model_params"]["random_state"] == 42
    assert payload["metrics"] == metrics


def test_save_feature_selection_artifacts_writes_files(monkeypatch, tmp_path):
    importance_path = tmp_path / "permutation_importance.csv"
    selected_features_path = tmp_path / "selected_features.json"

    monkeypatch.setattr(pipeline, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "PERMUTATION_IMPORTANCE_PATH", importance_path)
    monkeypatch.setattr(pipeline, "SELECTED_FEATURES_PATH", selected_features_path)

    importance_df = make_importance_df_with_random_probes()
    selected_features = FEATURE_COLUMNS[:8]
    metrics = {"valid_mae_before_selection": 1.23}
    best_params = {"max_iter": 100}

    pipeline.save_feature_selection_artifacts(
        importance_df=importance_df,
        selected_features=selected_features,
        metrics=metrics,
        best_params=best_params,
    )

    assert importance_path.exists()
    assert selected_features_path.exists()

    saved_payload = json.loads(selected_features_path.read_text(encoding="utf-8"))
    assert saved_payload["selected_features"] == selected_features
    assert (
        saved_payload["selection_method"] == "permutation_importance_with_random_probes"
    )
    assert saved_payload["random_probe_threshold"] == pytest.approx(0.10)
    assert saved_payload["metrics"] == metrics


def test_load_training_dataframe_uses_cache_when_requested(monkeypatch, tmp_path):
    cache_path = tmp_path / "temperature_features.parquet"
    df = make_training_df(periods=30)
    df.to_parquet(cache_path)

    monkeypatch.setattr(pipeline, "FEATURE_CACHE_PATH", cache_path)

    result = pipeline.load_training_dataframe(
        fs=None,
        use_cache=True,
    )

    assert len(result) == 30
    assert result["event_time"].is_monotonic_increasing
