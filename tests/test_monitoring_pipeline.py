from datetime import date, timedelta

import pandas as pd
import pytest

import monitoring_pipeline as pipeline


def make_prediction_log_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location_id": ["basel", "basel"],
            "feature_event_time": [
                "2026-05-01 00:00:00",
                "2026-05-01 01:00:00",
            ],
            "predicted_event_time": [
                "2026-05-01 06:00:00",
                "2026-05-01 07:00:00",
            ],
            "predicted_temperature_celsius": [10.0, 12.0],
            "model_version": ["1", "1"],
        }
    )


def make_actual_temperature_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location_id": ["basel", "basel"],
            "predicted_event_time": pd.to_datetime(
                [
                    "2026-05-01 06:00:00",
                    "2026-05-01 07:00:00",
                ]
            ),
            "actual_temperature_celsius": [11.5, 10.0],
        }
    )


def test_load_prediction_log_reads_valid_csv(monkeypatch, tmp_path):
    prediction_log_path = tmp_path / "prediction_log.csv"
    make_prediction_log_df().to_csv(prediction_log_path, index=False)

    monkeypatch.setattr(pipeline, "PREDICTION_LOG_PATH", prediction_log_path)

    result = pipeline.load_prediction_log()

    assert len(result) == 2
    assert result["feature_event_time"].dtype.kind == "M"
    assert result["predicted_event_time"].dtype.kind == "M"


def test_load_prediction_log_rejects_missing_file(monkeypatch, tmp_path):
    prediction_log_path = tmp_path / "missing_prediction_log.csv"

    monkeypatch.setattr(pipeline, "PREDICTION_LOG_PATH", prediction_log_path)

    with pytest.raises(FileNotFoundError, match="Prediction log not found"):
        pipeline.load_prediction_log()


def test_load_prediction_log_rejects_empty_csv(monkeypatch, tmp_path):
    prediction_log_path = tmp_path / "prediction_log.csv"

    pd.DataFrame(
        columns=[
            "location_id",
            "feature_event_time",
            "predicted_event_time",
            "predicted_temperature_celsius",
        ]
    ).to_csv(prediction_log_path, index=False)

    monkeypatch.setattr(pipeline, "PREDICTION_LOG_PATH", prediction_log_path)

    with pytest.raises(ValueError, match="Prediction log is empty"):
        pipeline.load_prediction_log()


def test_load_prediction_log_rejects_missing_columns(monkeypatch, tmp_path):
    prediction_log_path = tmp_path / "prediction_log.csv"

    pd.DataFrame(
        {
            "location_id": ["basel"],
            "predicted_temperature_celsius": [10.0],
        }
    ).to_csv(prediction_log_path, index=False)

    monkeypatch.setattr(pipeline, "PREDICTION_LOG_PATH", prediction_log_path)

    with pytest.raises(ValueError, match="Prediction log is missing columns"):
        pipeline.load_prediction_log()


def test_fetch_actual_temperatures_for_predictions(monkeypatch):
    prediction_log_df = make_prediction_log_df()
    prediction_log_df["predicted_event_time"] = pd.to_datetime(
        prediction_log_df["predicted_event_time"]
    )

    def fake_fetch_historical_weather(
        start_date,
        end_date,
    ):
        assert start_date == "2026-05-01"
        assert end_date == "2026-05-01"

        return pd.DataFrame(
            {
                "location_id": ["basel", "basel"],
                "event_time": pd.to_datetime(
                    [
                        "2026-05-01 06:00:00",
                        "2026-05-01 07:00:00",
                    ]
                ),
                "temperature_2m": [11.5, 10.0],
            }
        )

    monkeypatch.setattr(
        pipeline,
        "fetch_historical_weather",
        fake_fetch_historical_weather,
    )

    result = pipeline.fetch_actual_temperatures_for_predictions(prediction_log_df)

    assert result.columns.tolist() == [
        "location_id",
        "predicted_event_time",
        "actual_temperature_celsius",
    ]
    assert result["location_id"].tolist() == ["basel", "basel"]
    assert result["actual_temperature_celsius"].tolist() == [11.5, 10.0]


def test_fetch_actual_temperatures_rejects_empty_response(monkeypatch):
    prediction_log_df = make_prediction_log_df()
    prediction_log_df["predicted_event_time"] = pd.to_datetime(
        prediction_log_df["predicted_event_time"]
    )

    def fake_fetch_historical_weather(
        start_date,
        end_date,
    ):
        return pd.DataFrame()

    monkeypatch.setattr(
        pipeline,
        "fetch_historical_weather",
        fake_fetch_historical_weather,
    )

    with pytest.raises(ValueError, match="Historical weather response is empty"):
        pipeline.fetch_actual_temperatures_for_predictions(prediction_log_df)


def test_build_monitoring_results_computes_errors():
    prediction_log_df = make_prediction_log_df()
    prediction_log_df["feature_event_time"] = pd.to_datetime(
        prediction_log_df["feature_event_time"]
    )
    prediction_log_df["predicted_event_time"] = pd.to_datetime(
        prediction_log_df["predicted_event_time"]
    )

    actual_temperature_df = make_actual_temperature_df()

    result = pipeline.build_monitoring_results(
        prediction_log_df=prediction_log_df,
        actual_temperature_df=actual_temperature_df,
    )

    assert len(result) == 2
    assert result["error_celsius"].tolist() == [-1.5, 2.0]
    assert result["absolute_error_celsius"].tolist() == [1.5, 2.0]
    assert result["squared_error_celsius"].tolist() == [2.25, 4.0]
    assert result["rolling_mae_7_predictions"].tolist() == [1.5, 1.75]


def test_build_monitoring_results_rejects_unmatched_predictions():
    prediction_log_df = make_prediction_log_df()
    prediction_log_df["feature_event_time"] = pd.to_datetime(
        prediction_log_df["feature_event_time"]
    )
    prediction_log_df["predicted_event_time"] = pd.to_datetime(
        prediction_log_df["predicted_event_time"]
    )

    actual_temperature_df = pd.DataFrame(
        {
            "location_id": ["zurich"],
            "predicted_event_time": pd.to_datetime(["2026-05-01 06:00:00"]),
            "actual_temperature_celsius": [11.5],
        }
    )

    with pytest.raises(ValueError, match="No prediction rows could be matched"):
        pipeline.build_monitoring_results(
            prediction_log_df=prediction_log_df,
            actual_temperature_df=actual_temperature_df,
        )


def test_summarize_monitoring_results_returns_expected_metrics():
    prediction_log_df = make_prediction_log_df()
    prediction_log_df["feature_event_time"] = pd.to_datetime(
        prediction_log_df["feature_event_time"]
    )
    prediction_log_df["predicted_event_time"] = pd.to_datetime(
        prediction_log_df["predicted_event_time"]
    )

    monitoring_df = pipeline.build_monitoring_results(
        prediction_log_df=prediction_log_df,
        actual_temperature_df=make_actual_temperature_df(),
    )

    summary = pipeline.summarize_monitoring_results(monitoring_df)

    assert summary["n_matched_predictions"] == 2
    assert summary["monitoring_mae"] == pytest.approx(1.75)
    assert summary["monitoring_rmse"] == pytest.approx(((2.25 + 4.0) / 2) ** 0.5)
    assert summary["latest_absolute_error_celsius"] == pytest.approx(2.0)


def test_save_monitoring_results_writes_csv(monkeypatch, tmp_path):
    monitoring_results_path = tmp_path / "monitoring_results.csv"

    monkeypatch.setattr(pipeline, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(
        pipeline,
        "MONITORING_RESULTS_PATH",
        monitoring_results_path,
    )

    monitoring_df = pd.DataFrame(
        {
            "location_id": ["basel"],
            "predicted_temperature_celsius": [10.0],
            "actual_temperature_celsius": [11.0],
            "absolute_error_celsius": [1.0],
        }
    )

    pipeline.save_monitoring_results(monitoring_df)

    assert monitoring_results_path.exists()

    saved_df = pd.read_csv(monitoring_results_path)

    assert len(saved_df) == 1
    assert saved_df["absolute_error_celsius"].iloc[0] == 1.0


def test_fetch_actual_temperatures_rejects_future_predictions_only(monkeypatch):
    future_time = pd.Timestamp(date.today() + timedelta(days=1))

    prediction_log_df = pd.DataFrame(
        {
            "location_id": ["basel"],
            "feature_event_time": [future_time - pd.Timedelta(hours=6)],
            "predicted_event_time": [future_time],
            "predicted_temperature_celsius": [12.0],
        }
    )

    def fake_fetch_historical_weather(start_date, end_date):
        raise AssertionError("Historical weather should not be fetched.")

    monkeypatch.setattr(
        pipeline,
        "fetch_historical_weather",
        fake_fetch_historical_weather,
    )

    with pytest.raises(ValueError, match="No predictions are old enough"):
        pipeline.fetch_actual_temperatures_for_predictions(prediction_log_df)
