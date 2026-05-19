import json
from types import SimpleNamespace

import pandas as pd
import pytest

import tuning_pipeline as pipeline
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


def test_add_fixed_model_params_adds_reproducible_defaults():
    params = {"max_iter": 20}

    result = pipeline.add_fixed_model_params(params)

    assert result["max_iter"] == 20
    assert result["loss"] == "squared_error"
    assert result["early_stopping"] is False
    assert result["random_state"] == 42


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


def test_create_objective_returns_float_value():
    df = make_training_df(periods=80)
    train_df, valid_df, _test_df = pipeline.time_based_train_valid_test_split(df)

    X_train, y_train = pipeline.split_features_and_target(train_df)
    X_valid, y_valid = pipeline.split_features_and_target(valid_df)

    objective = pipeline.create_objective(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
    )

    class DummyTrial:
        def suggest_int(self, name, low, high):
            suggestions = {
                "max_iter": 20,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 10,
                "max_bins": 64,
            }
            return suggestions[name]

        def suggest_float(self, name, low, high, log=False):
            suggestions = {
                "learning_rate": 0.05,
                "l2_regularization": 0.01,
            }
            return suggestions[name]

    value = objective(DummyTrial())

    assert isinstance(value, float)
    assert value >= 0


def test_train_best_model_and_evaluate_returns_expected_metric_keys():
    df = make_training_df(periods=120)
    train_df, valid_df, test_df = pipeline.time_based_train_valid_test_split(df)

    fake_study = SimpleNamespace(
        best_params={
            "max_iter": 20,
            "learning_rate": 0.05,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 10,
            "l2_regularization": 0.01,
            "max_bins": 64,
        },
        best_value=1.23,
    )

    metrics = pipeline.train_best_model_and_evaluate(
        study=fake_study,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )

    assert set(metrics) == {
        "best_valid_mae",
        "tuned_test_mae",
        "tuned_test_rmse",
        "tuned_test_r2",
    }
    assert metrics["best_valid_mae"] == 1.23
    assert all(isinstance(value, float) for value in metrics.values())


def test_build_best_params_payload_contains_metadata():
    fake_study = SimpleNamespace(
        best_params={"max_iter": 100},
        best_value=1.23,
    )
    evaluation_metrics = {"tuned_test_mae": 1.5}

    payload = pipeline.build_best_params_payload(
        study=fake_study,
        evaluation_metrics=evaluation_metrics,
    )

    assert payload["model_type"] == "HistGradientBoostingRegressor"
    assert payload["best_params"] == {"max_iter": 100}
    assert payload["final_model_params"]["max_iter"] == 100
    assert payload["final_model_params"]["loss"] == "squared_error"
    assert payload["final_model_params"]["early_stopping"] is False
    assert payload["final_model_params"]["random_state"] == 42
    assert payload["best_valid_mae"] == 1.23
    assert payload["evaluation_metrics"] == evaluation_metrics
    assert payload["feature_columns"] == FEATURE_COLUMNS
    assert payload["target_column"] == TARGET_COLUMN


def test_save_tuning_artifacts_writes_files(monkeypatch, tmp_path):
    best_params_path = tmp_path / "best_params.json"
    tuning_results_path = tmp_path / "tuning_results.csv"

    fake_trials_df = pd.DataFrame(
        {
            "number": [0],
            "value": [1.23],
            "params_max_iter": [100],
        }
    )

    fake_study = SimpleNamespace(
        best_params={"max_iter": 100},
        best_value=1.23,
        trials_dataframe=lambda: fake_trials_df,
    )

    monkeypatch.setattr(pipeline, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "BEST_PARAMS_PATH", best_params_path)
    monkeypatch.setattr(pipeline, "TUNING_RESULTS_PATH", tuning_results_path)

    pipeline.save_tuning_artifacts(
        study=fake_study,
        evaluation_metrics={"tuned_test_mae": 1.5},
    )

    assert best_params_path.exists()
    assert tuning_results_path.exists()

    saved_payload = json.loads(best_params_path.read_text(encoding="utf-8"))
    assert saved_payload["best_params"] == {"max_iter": 100}
    assert saved_payload["final_model_params"]["loss"] == "squared_error"
    assert saved_payload["final_model_params"]["early_stopping"] is False
    assert saved_payload["final_model_params"]["random_state"] == 42
    assert saved_payload["evaluation_metrics"] == {"tuned_test_mae": 1.5}


def test_main_rejects_invalid_n_trials():
    with pytest.raises(ValueError, match="n_trials"):
        pipeline.main(n_trials=0)


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
