import pandas as pd
import pytest

import baseline_training_pipeline as pipeline
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


@pytest.mark.parametrize(
    ("train_fraction", "valid_fraction"),
    [
        (0.0, 0.15),
        (0.70, 0.0),
        (0.90, 0.10),
        (1.10, 0.10),
    ],
)
def test_time_based_train_valid_test_split_rejects_invalid_fractions(
    train_fraction,
    valid_fraction,
):
    df = make_training_df(periods=100)

    with pytest.raises(ValueError):
        pipeline.time_based_train_valid_test_split(
            df,
            train_fraction=train_fraction,
            valid_fraction=valid_fraction,
        )


def test_split_features_and_target_returns_expected_shapes():
    df = make_training_df(periods=30)

    X, y = pipeline.split_features_and_target(df)

    assert X.columns.tolist() == FEATURE_COLUMNS
    assert len(X) == 30
    assert len(y) == 30


def test_evaluate_naive_baselines_returns_expected_metric_keys():
    df = make_training_df(periods=30)
    X, y = pipeline.split_features_and_target(df)

    metrics = pipeline.evaluate_naive_baselines(X_test=X, y_test=y)

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


def test_train_model_can_fit_and_predict():
    df = make_training_df(periods=100)
    X, y = pipeline.split_features_and_target(df)

    model = pipeline.train_model(X_train=X, y_train=y)
    predictions = model.predict(X.head(5))

    assert len(predictions) == 5


def test_evaluate_model_returns_expected_metric_keys():
    df = make_training_df(periods=120)
    train_df, valid_df, test_df = pipeline.time_based_train_valid_test_split(df)

    X_train, y_train = pipeline.split_features_and_target(train_df)
    X_valid, y_valid = pipeline.split_features_and_target(valid_df)
    X_test, y_test = pipeline.split_features_and_target(test_df)

    model = pipeline.train_model(X_train=X_train, y_train=y_train)

    metrics = pipeline.evaluate_model(
        model=model,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
    )

    assert set(metrics) == {
        "valid_mae",
        "valid_rmse",
        "valid_r2",
        "test_mae",
        "test_rmse",
        "test_r2",
    }
    assert all(isinstance(value, float) for value in metrics.values())


def test_combine_metrics_adds_improvement_over_best_naive_baseline():
    model_metrics = {
        "test_mae": 2.0,
        "test_rmse": 3.0,
        "test_r2": 0.8,
        "valid_mae": 2.1,
        "valid_rmse": 3.1,
        "valid_r2": 0.7,
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

    metrics = pipeline.combine_metrics(
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
    )

    assert metrics["mae_improvement_over_best_naive_baseline"] == pytest.approx(0.5)


def test_build_model_bundle_contains_required_metadata():
    class DummyModel:
        def get_params(self):
            return {
                "max_iter": 200,
                "learning_rate": 0.05,
                "max_leaf_nodes": 31,
            }

    dummy_model = DummyModel()
    metrics = {"test_mae": 1.23}

    bundle = pipeline.build_model_bundle(
        model=dummy_model,
        metrics=metrics,
    )

    assert bundle["model"] is dummy_model
    assert bundle["feature_columns"] == FEATURE_COLUMNS
    assert bundle["target_column"] == TARGET_COLUMN
    assert bundle["metrics"] == metrics
    assert bundle["model_type"] == "HistGradientBoostingRegressor"
    assert bundle["model_params"] == {
        "max_iter": 200,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
    }


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


def test_main_skips_model_upload_when_requested(monkeypatch, tmp_path):
    df = make_training_df(periods=120)
    cache_path = tmp_path / "temperature_features.parquet"
    df.to_parquet(cache_path)

    monkeypatch.setattr(pipeline, "FEATURE_CACHE_PATH", cache_path)
    monkeypatch.setattr(pipeline, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(
        pipeline, "MODEL_LOCAL_PATH", tmp_path / "models" / "model.joblib"
    )

    upload_called = False

    def fake_upload_model_to_registry(metrics):
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
