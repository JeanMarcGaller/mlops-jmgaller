import json

import pandas as pd
import pytest

import inference_pipeline as pipeline
from config import TARGET_COLUMN


class DummyModel:
    def predict(self, X):
        return [12.34]


def test_get_latest_model_version_returns_highest_version():
    class FakeModel:
        def __init__(self, version):
            self.version = version

    class FakeModelRegistry:
        def get_models(self, name):
            return [
                FakeModel(version=1),
                FakeModel(version=3),
                FakeModel(version=2),
            ]

    result = pipeline.get_latest_model_version(
        model_registry=FakeModelRegistry(),
        model_name="temperature_forecast_regressor",
    )

    assert result == 3


def test_get_latest_model_version_rejects_empty_registry_result():
    class FakeModelRegistry:
        def get_models(self, name):
            return []

    with pytest.raises(ValueError, match="No models found"):
        pipeline.get_latest_model_version(
            model_registry=FakeModelRegistry(),
            model_name="temperature_forecast_regressor",
        )


def test_normalize_feature_vector_keeps_dataframe():
    df = pd.DataFrame(
        {
            "location_id": ["basel"],
            "event_time_unix": [1],
            "temperature_2m": [10.0],
        }
    )

    result = pipeline.normalize_feature_vector(
        feature_vector=df,
        expected_columns=df.columns.tolist(),
    )

    assert result.equals(df)


def test_normalize_feature_vector_converts_dict():
    feature_vector = {
        "location_id": "basel",
        "event_time_unix": 1,
        "temperature_2m": 10.0,
    }

    result = pipeline.normalize_feature_vector(
        feature_vector=feature_vector,
        expected_columns=list(feature_vector.keys()),
    )

    assert result.shape == (1, 3)
    assert result.loc[0, "location_id"] == "basel"
    assert result.loc[0, "temperature_2m"] == 10.0


def test_normalize_feature_vector_restores_expected_columns_for_sequence():
    expected_columns = [
        "location_id",
        "event_time_unix",
        "temperature_2m",
    ]

    feature_vector = ["basel", 1, 10.0]

    result = pipeline.normalize_feature_vector(
        feature_vector=feature_vector,
        expected_columns=expected_columns,
    )

    assert result.columns.tolist() == expected_columns
    assert result.loc[0, "location_id"] == "basel"


def test_prepare_model_input_returns_columns_in_model_order():
    feature_vector_df = pd.DataFrame(
        {
            "temperature_2m": [10.0],
            "hour": [12],
            "pressure_msl": [1010.0],
        }
    )

    feature_columns = [
        "hour",
        "temperature_2m",
    ]

    result = pipeline.prepare_model_input(
        feature_vector_df=feature_vector_df,
        feature_columns=feature_columns,
    )

    assert result.columns.tolist() == feature_columns
    assert result.shape == (1, 2)


def test_prepare_model_input_rejects_missing_columns():
    feature_vector_df = pd.DataFrame(
        {
            "temperature_2m": [10.0],
        }
    )

    with pytest.raises(ValueError, match="Missing model input columns"):
        pipeline.prepare_model_input(
            feature_vector_df=feature_vector_df,
            feature_columns=["temperature_2m", "hour"],
        )


def test_predict_temperature_returns_float():
    X = pd.DataFrame({"temperature_2m": [10.0]})

    result = pipeline.predict_temperature(
        model=DummyModel(),
        X=X,
    )

    assert result == 12.34
    assert isinstance(result, float)


def test_validate_model_bundle_accepts_required_keys():
    model_bundle = {
        "model": DummyModel(),
        "feature_columns": ["temperature_2m"],
    }

    pipeline.validate_model_bundle(model_bundle)


def test_validate_model_bundle_rejects_missing_model():
    model_bundle = {
        "feature_columns": ["temperature_2m"],
    }

    with pytest.raises(ValueError, match="Model bundle is missing"):
        pipeline.validate_model_bundle(model_bundle)


def test_validate_model_bundle_rejects_missing_feature_columns():
    model_bundle = {
        "model": DummyModel(),
    }

    with pytest.raises(ValueError, match="Model bundle is missing"):
        pipeline.validate_model_bundle(model_bundle)


def test_build_prediction_record_returns_serializable_metadata():
    metrics = {
        "test_mae": 1.23,
        "test_rmse": 2.34,
        "mae_improvement_over_best_naive_baseline": 0.56,
    }

    record = pipeline.build_prediction_record(
        location_id="basel",
        feature_event_time=pd.Timestamp("2026-05-01 00:00:00"),
        predicted_event_time=pd.Timestamp("2026-05-01 06:00:00"),
        target_column=TARGET_COLUMN,
        predicted_temperature=12.345,
        model_metrics=metrics,
        model_version=7,
    )

    assert record == {
        "location_id": "basel",
        "feature_event_time": "2026-05-01 00:00:00",
        "predicted_event_time": "2026-05-01 06:00:00",
        "target_column": TARGET_COLUMN,
        "predicted_temperature_celsius": 12.345,
        "model_version": "7",
        "model_test_mae": 1.23,
        "model_test_rmse": 2.34,
        "mae_improvement_over_best_naive_baseline": 0.56,
    }


def test_save_latest_prediction_writes_json(monkeypatch, tmp_path):
    latest_prediction_path = tmp_path / "latest_prediction.json"

    monkeypatch.setattr(pipeline, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(
        pipeline,
        "LATEST_PREDICTION_PATH",
        latest_prediction_path,
    )

    prediction_record = {
        "location_id": "basel",
        "predicted_temperature_celsius": 12.3,
    }

    pipeline.save_latest_prediction(prediction_record)

    saved_record = json.loads(latest_prediction_path.read_text(encoding="utf-8"))

    assert saved_record == prediction_record


def test_append_prediction_log_creates_csv(monkeypatch, tmp_path):
    prediction_log_path = tmp_path / "prediction_log.csv"

    monkeypatch.setattr(pipeline, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(
        pipeline,
        "PREDICTION_LOG_PATH",
        prediction_log_path,
    )

    prediction_record = {
        "location_id": "basel",
        "predicted_temperature_celsius": 12.3,
    }

    pipeline.append_prediction_log(prediction_record)

    saved_df = pd.read_csv(prediction_log_path)

    assert len(saved_df) == 1
    assert saved_df["location_id"].iloc[0] == "basel"
    assert saved_df["predicted_temperature_celsius"].iloc[0] == 12.3


def test_append_prediction_log_appends_to_existing_csv(monkeypatch, tmp_path):
    prediction_log_path = tmp_path / "prediction_log.csv"

    monkeypatch.setattr(pipeline, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(
        pipeline,
        "PREDICTION_LOG_PATH",
        prediction_log_path,
    )

    first_record = {
        "location_id": "basel",
        "predicted_temperature_celsius": 12.3,
    }
    second_record = {
        "location_id": "basel",
        "predicted_temperature_celsius": 13.4,
    }

    pipeline.append_prediction_log(first_record)
    pipeline.append_prediction_log(second_record)

    saved_df = pd.read_csv(prediction_log_path)

    assert len(saved_df) == 2
    assert saved_df["predicted_temperature_celsius"].tolist() == [12.3, 13.4]


def test_save_prediction_artifacts_writes_latest_and_log(monkeypatch, tmp_path):
    latest_prediction_path = tmp_path / "latest_prediction.json"
    prediction_log_path = tmp_path / "prediction_log.csv"

    monkeypatch.setattr(pipeline, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(
        pipeline,
        "LATEST_PREDICTION_PATH",
        latest_prediction_path,
    )
    monkeypatch.setattr(
        pipeline,
        "PREDICTION_LOG_PATH",
        prediction_log_path,
    )

    prediction_record = {
        "location_id": "basel",
        "predicted_temperature_celsius": 12.3,
    }

    pipeline.save_prediction_artifacts(prediction_record)

    assert latest_prediction_path.exists()
    assert prediction_log_path.exists()
