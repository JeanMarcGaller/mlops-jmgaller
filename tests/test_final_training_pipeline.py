import json

import pandas as pd
import pytest

import final_training_pipeline as pipeline
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


def test_load_best_params_reads_valid_json(monkeypatch, tmp_path):
    best_params_path = tmp_path / "best_params.json"
    best_params = {
        "max_iter": 20,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 10,
        "l2_regularization": 0.01,
        "max_bins": 64,
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


def test_load_selected_features_reads_valid_json(monkeypatch, tmp_path):
    selected_features_path = tmp_path / "selected_features.json"
    selected_features = FEATURE_COLUMNS[:8]

    selected_features_path.write_text(
        json.dumps({"selected_features": selected_features}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        pipeline,
        "SELECTED_FEATURES_PATH",
        selected_features_path,
    )

    result = pipeline.load_selected_features()

    assert result == selected_features


def test_load_selected_features_rejects_missing_file(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing_selected_features.json"
    monkeypatch.setattr(pipeline, "SELECTED_FEATURES_PATH", missing_path)

    with pytest.raises(FileNotFoundError, match="Selected features file not found"):
        pipeline.load_selected_features()


def test_load_selected_features_rejects_missing_selected_features_key(
    monkeypatch,
    tmp_path,
):
    selected_features_path = tmp_path / "selected_features.json"
    selected_features_path.write_text(json.dumps({"other_key": []}), encoding="utf-8")

    monkeypatch.setattr(
        pipeline,
        "SELECTED_FEATURES_PATH",
        selected_features_path,
    )

    with pytest.raises(ValueError, match="No 'selected_features'"):
        pipeline.load_selected_features()


def test_validate_training_dataframe_sorts_by_event_time():
    selected_features = FEATURE_COLUMNS[:8]
    df = make_training_df(periods=30)
    shuffled_df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    result = pipeline.validate_training_dataframe(
        df=shuffled_df,
        selected_features=selected_features,
    )

    assert result["event_time"].is_monotonic_increasing


def test_validate_training_dataframe_rejects_missing_target():
    selected_features = FEATURE_COLUMNS[:8]
    df = make_training_df(periods=30).drop(columns=[TARGET_COLUMN])

    with pytest.raises(ValueError, match="Missing target column"):
        pipeline.validate_training_dataframe(
            df=df,
            selected_features=selected_features,
        )


def test_validate_training_dataframe_rejects_missing_selected_feature():
    selected_features = FEATURE_COLUMNS[:8]
    df = make_training_df(periods=30).drop(columns=[selected_features[0]])

    with pytest.raises(ValueError, match="Missing selected feature columns"):
        pipeline.validate_training_dataframe(
            df=df,
            selected_features=selected_features,
        )


def test_validate_training_dataframe_rejects_missing_event_time():
    selected_features = FEATURE_COLUMNS[:8]
    df = make_training_df(periods=30).drop(columns=["event_time"])

    with pytest.raises(ValueError, match="Missing event_time"):
        pipeline.validate_training_dataframe(
            df=df,
            selected_features=selected_features,
        )


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


def test_split_features_and_target_uses_selected_features():
    df = make_training_df(periods=30)
    selected_features = FEATURE_COLUMNS[:8]

    X, y = pipeline.split_features_and_target(
        df=df,
        feature_columns=selected_features,
    )

    assert X.columns.tolist() == selected_features
    assert len(X) == 30
    assert len(y) == 30


def test_train_final_model_can_fit_and_predict():
    df = make_training_df(periods=120)
    selected_features = FEATURE_COLUMNS[:8]
    train_df, valid_df, test_df = pipeline.time_based_train_valid_test_split(df)

    best_params = {
        "max_iter": 20,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 10,
        "l2_regularization": 0.01,
        "max_bins": 64,
    }

    model = pipeline.train_final_model(
        train_df=train_df,
        valid_df=valid_df,
        selected_features=selected_features,
        best_params=best_params,
    )

    X_test, _y_test = pipeline.split_features_and_target(
        df=test_df,
        feature_columns=selected_features,
    )
    predictions = model.predict(X_test.head(5))

    assert len(predictions) == 5


def test_evaluate_naive_baselines_returns_expected_metric_keys():
    df = make_training_df(periods=30)

    metrics = pipeline.evaluate_naive_baselines(test_df=df)

    assert set(metrics) == {
        "persistence_mae",
        "persistence_rmse",
        "persistence_r2",
        "seasonal_naive_24h_mae",
        "seasonal_naive_24h_rmse",
        "seasonal_naive_24h_r2",
        "best_naive_baseline_mae",
    }
    assert all(isinstance(value, float) for value in metrics.values())


def test_evaluate_final_model_returns_expected_metric_keys():
    df = make_training_df(periods=120)
    selected_features = FEATURE_COLUMNS[:8]
    train_df, valid_df, test_df = pipeline.time_based_train_valid_test_split(df)

    best_params = {
        "max_iter": 20,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 10,
        "l2_regularization": 0.01,
        "max_bins": 64,
    }

    model = pipeline.train_final_model(
        train_df=train_df,
        valid_df=valid_df,
        selected_features=selected_features,
        best_params=best_params,
    )

    metrics = pipeline.evaluate_final_model(
        model=model,
        test_df=test_df,
        selected_features=selected_features,
    )

    assert set(metrics) == {
        "test_mae",
        "test_rmse",
        "test_r2",
    }
    assert all(isinstance(value, float) for value in metrics.values())


def test_combine_metrics_adds_selected_feature_count_and_improvement():
    final_metrics = {
        "test_mae": 2.0,
        "test_rmse": 3.0,
        "test_r2": 0.8,
    }
    baseline_metrics = {
        "persistence_mae": 5.0,
        "persistence_rmse": 6.0,
        "persistence_r2": -0.1,
        "seasonal_naive_24h_mae": 4.0,
        "seasonal_naive_24h_rmse": 5.0,
        "seasonal_naive_24h_r2": 0.1,
        "best_naive_baseline_mae": 4.0,
    }
    selected_features = FEATURE_COLUMNS[:8]

    metrics = pipeline.combine_metrics(
        final_metrics=final_metrics,
        baseline_metrics=baseline_metrics,
        selected_features=selected_features,
    )

    assert metrics["n_selected_features"] == 8
    assert metrics["mae_improvement_over_best_naive_baseline"] == pytest.approx(0.5)


def test_build_model_bundle_contains_required_metadata():
    class DummyModel:
        def get_params(self):
            return {
                "loss": "squared_error",
                "max_iter": 20,
                "early_stopping": False,
                "random_state": 42,
            }

    dummy_model = DummyModel()
    selected_features = FEATURE_COLUMNS[:8]
    best_params = {"max_iter": 20}
    metrics = {"test_mae": 1.23}

    bundle = pipeline.build_model_bundle(
        model=dummy_model,
        selected_features=selected_features,
        best_params=best_params,
        metrics=metrics,
    )

    assert bundle["model"] is dummy_model
    assert bundle["feature_columns"] == selected_features
    assert bundle["target_column"] == TARGET_COLUMN
    assert bundle["metrics"] == metrics
    assert bundle["model_type"] == "HistGradientBoostingRegressor"
    assert bundle["model_params"] == {
        "loss": "squared_error",
        "max_iter": 20,
        "early_stopping": False,
        "random_state": 42,
    }
    assert bundle["best_params"] == best_params
    assert bundle["feature_selection"] == "permutation_importance"


def test_add_fixed_model_params_adds_reproducible_defaults():
    params = {"max_iter": 20}

    result = pipeline.add_fixed_model_params(params)

    assert result["max_iter"] == 20
    assert result["loss"] == "squared_error"
    assert result["early_stopping"] is False
    assert result["random_state"] == 42


def test_main_skips_model_upload_when_requested(monkeypatch, tmp_path):
    df = make_training_df(periods=120)
    cache_path = tmp_path / "temperature_features.parquet"
    df.to_parquet(cache_path)

    best_params_path = tmp_path / "best_params.json"
    selected_features_path = tmp_path / "selected_features.json"

    selected_features = FEATURE_COLUMNS[:8]
    best_params = {
        "max_iter": 20,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 10,
        "l2_regularization": 0.01,
        "max_bins": 64,
    }

    best_params_path.write_text(
        json.dumps({"best_params": best_params}),
        encoding="utf-8",
    )
    selected_features_path.write_text(
        json.dumps({"selected_features": selected_features}),
        encoding="utf-8",
    )

    monkeypatch.setattr(pipeline, "FEATURE_CACHE_PATH", cache_path)
    monkeypatch.setattr(pipeline, "BEST_PARAMS_PATH", best_params_path)
    monkeypatch.setattr(pipeline, "SELECTED_FEATURES_PATH", selected_features_path)
    monkeypatch.setattr(pipeline, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(
        pipeline, "MODEL_LOCAL_PATH", tmp_path / "models" / "model.joblib"
    )

    upload_called = False

    def fake_upload_model_to_registry(selected_features, metrics):
        nonlocal upload_called
        upload_called = True

    monkeypatch.setattr(
        pipeline,
        "upload_model_to_registry",
        fake_upload_model_to_registry,
    )

    pipeline.main(
        use_cache=True,
        skip_upload=True,
    )

    assert upload_called is False
