import pandas as pd
import pytest

import model_comparison_pipeline as pipeline
from features import FEATURE_COLUMNS


class DummyMeanModel:
    def fit(self, X, y):
        self.mean_ = float(y.mean())
        return self

    def predict(self, X):
        return [self.mean_] * len(X)


class FailingModel:
    def fit(self, X, y):
        raise RuntimeError("boom")


def make_training_df(periods: int = 120) -> pd.DataFrame:
    event_time = pd.date_range(
        "2026-01-01 00:00:00",
        periods=periods,
        freq="h",
    )

    df = pd.DataFrame(
        {
            "event_time": event_time,
            "temperature_2m_next_6h": range(periods),
        }
    )

    for index, feature in enumerate(FEATURE_COLUMNS):
        df[feature] = index + 1

    return df


def test_validate_training_dataframe_sorts_by_event_time():
    df = make_training_df(periods=10).sample(frac=1, random_state=42)

    result = pipeline.validate_training_dataframe(df)

    assert result["event_time"].is_monotonic_increasing


def test_validate_training_dataframe_rejects_missing_target():
    df = make_training_df(periods=10).drop(columns=["temperature_2m_next_6h"])

    with pytest.raises(ValueError, match="Missing target column"):
        pipeline.validate_training_dataframe(df)


def test_validate_training_dataframe_rejects_missing_feature():
    df = make_training_df(periods=10).drop(columns=[FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match="Missing feature columns"):
        pipeline.validate_training_dataframe(df)


def test_validate_training_dataframe_rejects_missing_event_time():
    df = make_training_df(periods=10).drop(columns=["event_time"])

    with pytest.raises(ValueError, match="Missing event_time column"):
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


def test_time_based_train_valid_test_split_rejects_invalid_fractions():
    df = make_training_df(periods=100)

    with pytest.raises(ValueError, match="train_fraction"):
        pipeline.time_based_train_valid_test_split(df, train_fraction=0)

    with pytest.raises(ValueError, match="valid_fraction"):
        pipeline.time_based_train_valid_test_split(df, valid_fraction=0)

    with pytest.raises(ValueError, match="must be smaller than 1"):
        pipeline.time_based_train_valid_test_split(
            df,
            train_fraction=0.90,
            valid_fraction=0.20,
        )


def test_split_features_and_target_returns_expected_shapes():
    df = make_training_df(periods=20)

    X, y = pipeline.split_features_and_target(df, FEATURE_COLUMNS)

    assert X.columns.tolist() == FEATURE_COLUMNS
    assert len(X) == 20
    assert len(y) == 20


def test_build_model_candidates_contains_expected_models():
    candidates = pipeline.build_model_candidates()

    candidate_names = [name for name, _model in candidates]

    assert "HistGradientBoostingRegressor" in candidate_names
    assert "RandomForestRegressor" in candidate_names
    assert "XGBRegressor" in candidate_names


def test_compute_regression_metrics_returns_prefixed_keys():
    metrics = pipeline.compute_regression_metrics(
        y_true=[1.0, 2.0, 3.0],
        y_pred=[1.0, 2.5, 2.5],
        prefix="valid",
    )

    assert set(metrics) == {
        "valid_mae",
        "valid_rmse",
        "valid_r2",
    }
    assert metrics["valid_mae"] == pytest.approx((0.0 + 0.5 + 0.5) / 3)


def test_evaluate_model_candidate_returns_metrics_for_successful_model():
    df = make_training_df(periods=60)
    train_df, valid_df, test_df = pipeline.time_based_train_valid_test_split(df)

    X_train, y_train = pipeline.split_features_and_target(train_df, FEATURE_COLUMNS)
    X_valid, y_valid = pipeline.split_features_and_target(valid_df, FEATURE_COLUMNS)
    X_test, y_test = pipeline.split_features_and_target(test_df, FEATURE_COLUMNS)

    result = pipeline.evaluate_model_candidate(
        model_name="DummyMeanModel",
        model=DummyMeanModel(),
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
    )

    assert result["model_name"] == "DummyMeanModel"
    assert result["status"] == "ok"
    assert result["error_message"] == ""
    assert result["fit_seconds"] >= 0
    assert result["n_features"] == len(FEATURE_COLUMNS)
    assert result["valid_mae"] is not None
    assert result["test_mae"] is not None


def test_evaluate_model_candidate_marks_missing_model_as_skipped():
    df = make_training_df(periods=60)
    train_df, valid_df, test_df = pipeline.time_based_train_valid_test_split(df)

    X_train, y_train = pipeline.split_features_and_target(train_df, FEATURE_COLUMNS)
    X_valid, y_valid = pipeline.split_features_and_target(valid_df, FEATURE_COLUMNS)
    X_test, y_test = pipeline.split_features_and_target(test_df, FEATURE_COLUMNS)

    result = pipeline.evaluate_model_candidate(
        model_name="MissingModel",
        model=None,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
    )

    assert result["model_name"] == "MissingModel"
    assert result["status"] == "skipped"
    assert result["error_message"] == "Model package is not installed."
    assert result["test_mae"] is None


def test_evaluate_model_candidate_marks_failed_model_as_failed():
    df = make_training_df(periods=60)
    train_df, valid_df, test_df = pipeline.time_based_train_valid_test_split(df)

    X_train, y_train = pipeline.split_features_and_target(train_df, FEATURE_COLUMNS)
    X_valid, y_valid = pipeline.split_features_and_target(valid_df, FEATURE_COLUMNS)
    X_test, y_test = pipeline.split_features_and_target(test_df, FEATURE_COLUMNS)

    result = pipeline.evaluate_model_candidate(
        model_name="FailingModel",
        model=FailingModel(),
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
    )

    assert result["model_name"] == "FailingModel"
    assert result["status"] == "failed"
    assert "RuntimeError: boom" in result["error_message"]


def test_run_model_comparison_returns_results(monkeypatch):
    df = make_training_df(periods=80)
    train_df, valid_df, test_df = pipeline.time_based_train_valid_test_split(df)

    monkeypatch.setattr(
        pipeline,
        "build_model_candidates",
        lambda: [
            ("DummyMeanModel", DummyMeanModel()),
            ("MissingModel", None),
        ],
    )

    result = pipeline.run_model_comparison(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        feature_columns=FEATURE_COLUMNS,
    )

    assert result["model_name"].tolist() == [
        "DummyMeanModel",
        "MissingModel",
    ]
    assert result["status"].tolist() == [
        "ok",
        "skipped",
    ]


def test_save_model_comparison_results_writes_csv(monkeypatch, tmp_path):
    model_comparison_results_path = tmp_path / "model_comparison_results.csv"

    monkeypatch.setattr(pipeline, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(
        pipeline,
        "MODEL_COMPARISON_RESULTS_PATH",
        model_comparison_results_path,
    )

    comparison_df = pd.DataFrame(
        {
            "model_name": ["DummyMeanModel"],
            "status": ["ok"],
            "test_mae": [1.23],
        }
    )

    pipeline.save_model_comparison_results(comparison_df)

    assert model_comparison_results_path.exists()

    saved_df = pd.read_csv(model_comparison_results_path)

    assert len(saved_df) == 1
    assert saved_df["model_name"].iloc[0] == "DummyMeanModel"
    assert saved_df["test_mae"].iloc[0] == 1.23
