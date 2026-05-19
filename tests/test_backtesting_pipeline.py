import pandas as pd
import pytest

import backtesting_pipeline as pipeline
from config import TARGET_COLUMN
from features import FEATURE_COLUMNS


def make_training_df(periods: int = 120) -> pd.DataFrame:
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


def test_load_training_dataframe_uses_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "temperature_features.parquet"
    df = make_training_df(periods=30)
    df.to_parquet(cache_path)

    monkeypatch.setattr(pipeline, "FEATURE_CACHE_PATH", cache_path)

    result = pipeline.load_training_dataframe(use_cache=True)

    assert len(result) == 30
    assert result["event_time"].is_monotonic_increasing


def test_load_training_dataframe_rejects_non_cache_mode():
    with pytest.raises(NotImplementedError, match="only --use-cache"):
        pipeline.load_training_dataframe(use_cache=False)


def test_create_backtest_windows_returns_expected_number_of_windows():
    df = make_training_df(periods=120)

    windows = pipeline.create_backtest_windows(
        df=df,
        n_splits=4,
    )

    assert len(windows) == 4

    expected_train_lengths = [60, 72, 84, 96]

    for index, (train_df, valid_df, test_df) in enumerate(windows):
        assert len(train_df) == expected_train_lengths[index]
        assert len(valid_df) == 12
        assert len(test_df) == 12

        assert train_df["event_time"].min() == df["event_time"].min()
        assert train_df["event_time"].max() < valid_df["event_time"].min()
        assert valid_df["event_time"].max() < test_df["event_time"].min()

    assert windows[0][0]["event_time"].max() < windows[1][0]["event_time"].max()
    assert windows[1][0]["event_time"].max() < windows[2][0]["event_time"].max()
    assert windows[2][0]["event_time"].max() < windows[3][0]["event_time"].max()


@pytest.mark.parametrize(
    "n_splits",
    [
        0,
        -1,
    ],
)
def test_create_backtest_windows_rejects_invalid_n_splits(n_splits):
    df = make_training_df(periods=120)

    with pytest.raises(ValueError, match="n_splits"):
        pipeline.create_backtest_windows(
            df=df,
            n_splits=n_splits,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"initial_train_fraction": 0}, "initial_train_fraction"),
        ({"initial_train_fraction": 1}, "initial_train_fraction"),
        ({"valid_fraction": 0}, "valid_fraction"),
        ({"valid_fraction": 1}, "valid_fraction"),
        ({"test_fraction": 0}, "test_fraction"),
        ({"test_fraction": 1}, "test_fraction"),
    ],
)
def test_create_backtest_windows_rejects_invalid_fractions(kwargs, match):
    df = make_training_df(periods=120)

    with pytest.raises(ValueError, match=match):
        pipeline.create_backtest_windows(
            df=df,
            n_splits=4,
            **kwargs,
        )


def test_create_backtest_windows_rejects_too_many_splits():
    df = make_training_df(periods=120)

    with pytest.raises(ValueError, match="Dataset is too small"):
        pipeline.create_backtest_windows(
            df=df,
            n_splits=10,
            initial_train_fraction=0.50,
            valid_fraction=0.10,
            test_fraction=0.10,
        )


def test_split_features_and_target_returns_expected_shapes():
    df = make_training_df(periods=30)

    X, y = pipeline.split_features_and_target(
        df=df,
        feature_columns=FEATURE_COLUMNS,
    )

    assert X.columns.tolist() == FEATURE_COLUMNS
    assert len(X) == 30
    assert len(y) == 30


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


def test_train_model_can_fit_and_predict():
    df = make_training_df(periods=80)
    X, y = pipeline.split_features_and_target(
        df=df,
        feature_columns=FEATURE_COLUMNS,
    )

    model = pipeline.train_model(
        X_train=X,
        y_train=y,
    )

    predictions = model.predict(X.head(5))

    assert len(predictions) == 5


def test_evaluate_model_returns_expected_metric_keys():
    df = make_training_df(periods=80)
    X, y = pipeline.split_features_and_target(
        df=df,
        feature_columns=FEATURE_COLUMNS,
    )

    model = pipeline.train_model(
        X_train=X,
        y_train=y,
    )

    metrics = pipeline.evaluate_model(
        model=model,
        X_test=X,
        y_test=y,
    )

    assert set(metrics) == {
        "model_mae",
        "model_rmse",
        "model_r2",
    }
    assert all(isinstance(value, float) for value in metrics.values())


def test_evaluate_backtest_window_returns_expected_metadata_and_metrics():
    df = make_training_df(periods=120)
    train_df, valid_df, test_df = pipeline.create_backtest_windows(
        df=df,
        n_splits=4,
    )[0]

    result = pipeline.evaluate_backtest_window(
        split_id=1,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )

    expected_keys = {
        "split_id",
        "train_start",
        "train_end",
        "valid_start",
        "valid_end",
        "test_start",
        "test_end",
        "n_train",
        "n_valid",
        "n_test",
        "persistence_mae",
        "persistence_rmse",
        "persistence_r2",
        "seasonal_naive_24h_mae",
        "seasonal_naive_24h_rmse",
        "seasonal_naive_24h_r2",
        "best_naive_baseline_mae",
        "model_mae",
        "model_rmse",
        "model_r2",
        "mae_improvement_over_best_naive_baseline",
    }

    assert set(result) == expected_keys
    assert result["split_id"] == 1
    assert result["n_train"] == len(train_df)
    assert result["n_valid"] == len(valid_df)
    assert result["n_test"] == len(test_df)


def test_run_backtesting_returns_one_row_per_split():
    df = make_training_df(periods=120)

    result = pipeline.run_backtesting(
        df=df,
        n_splits=4,
    )

    assert len(result) == 4
    assert result["split_id"].tolist() == [1, 2, 3, 4]
    assert "model_mae" in result.columns
    assert "best_naive_baseline_mae" in result.columns


def test_save_backtesting_results_writes_csv(monkeypatch, tmp_path):
    results_path = tmp_path / "backtesting_results.csv"

    monkeypatch.setattr(pipeline, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "BACKTESTING_RESULTS_PATH", results_path)

    results_df = pd.DataFrame(
        {
            "split_id": [1, 2],
            "model_mae": [1.1, 1.2],
            "best_naive_baseline_mae": [3.0, 3.1],
        }
    )

    pipeline.save_backtesting_results(results_df)

    assert results_path.exists()

    saved_df = pd.read_csv(results_path)

    assert saved_df["split_id"].tolist() == [1, 2]
    assert saved_df["model_mae"].tolist() == [1.1, 1.2]
