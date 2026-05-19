"""Monitoring pipeline for logged temperature predictions.

This pipeline reads local prediction logs, fetches actual historical
temperatures for the predicted timestamps and computes forecast errors.

It is intentionally report-based and local-first:
- input: reports/prediction_log.csv
- output: reports/monitoring_results.csv
"""

import argparse
from datetime import date, timedelta

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import (
    MONITORING_RESULTS_PATH,
    PREDICTION_LOG_PATH,
    REPORTS_DIR,
)
from weather_api import fetch_historical_weather


def load_prediction_log() -> pd.DataFrame:
    """Load local prediction log."""
    if not PREDICTION_LOG_PATH.exists():
        raise FileNotFoundError(
            f"Prediction log not found at {PREDICTION_LOG_PATH}. "
            "Run inference_pipeline.py first."
        )

    prediction_log_df = pd.read_csv(PREDICTION_LOG_PATH)

    if prediction_log_df.empty:
        raise ValueError(f"Prediction log is empty: {PREDICTION_LOG_PATH}")

    required_columns = {
        "location_id",
        "feature_event_time",
        "predicted_event_time",
        "predicted_temperature_celsius",
    }

    missing_columns = required_columns - set(prediction_log_df.columns)

    if missing_columns:
        raise ValueError(
            f"Prediction log is missing columns: {sorted(missing_columns)}"
        )

    prediction_log_df = prediction_log_df.copy()
    prediction_log_df["feature_event_time"] = pd.to_datetime(
        prediction_log_df["feature_event_time"]
    )
    prediction_log_df["predicted_event_time"] = pd.to_datetime(
        prediction_log_df["predicted_event_time"]
    )

    return prediction_log_df


def fetch_actual_temperatures_for_predictions(
    prediction_log_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch actual temperatures for all predicted timestamps."""
    min_predicted_time = prediction_log_df["predicted_event_time"].min()
    max_predicted_time = prediction_log_df["predicted_event_time"].max()

    latest_available_date = date.today() - timedelta(days=1)

    available_prediction_log_df = prediction_log_df.loc[
        prediction_log_df["predicted_event_time"].dt.date <= latest_available_date
    ].copy()

    if available_prediction_log_df.empty:
        raise ValueError(
            "No predictions are old enough for monitoring yet. "
            f"Latest available historical date is {latest_available_date}."
        )

    min_predicted_time = available_prediction_log_df["predicted_event_time"].min()
    max_predicted_time = available_prediction_log_df["predicted_event_time"].max()

    start_date = min_predicted_time.date().isoformat()
    end_date = max_predicted_time.date().isoformat()

    actual_weather_df = fetch_historical_weather(
        start_date=start_date,
        end_date=end_date,
    )

    if actual_weather_df.empty:
        raise ValueError("Historical weather response is empty.")

    actual_weather_df = actual_weather_df.copy()
    actual_weather_df["event_time"] = pd.to_datetime(actual_weather_df["event_time"])

    actual_temperature_df = actual_weather_df[
        [
            "location_id",
            "event_time",
            "temperature_2m",
        ]
    ].rename(
        columns={
            "event_time": "predicted_event_time",
            "temperature_2m": "actual_temperature_celsius",
        }
    )

    return actual_temperature_df


def build_monitoring_results(
    prediction_log_df: pd.DataFrame,
    actual_temperature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join predictions with actual temperatures and compute errors."""
    monitoring_df = prediction_log_df.merge(
        actual_temperature_df,
        on=["location_id", "predicted_event_time"],
        how="left",
    )

    monitoring_df = monitoring_df.dropna(subset=["actual_temperature_celsius"]).copy()

    if monitoring_df.empty:
        raise ValueError(
            "No prediction rows could be matched with actual temperatures. "
            "The predicted timestamps may still be in the future."
        )

    monitoring_df["error_celsius"] = (
        monitoring_df["predicted_temperature_celsius"]
        - monitoring_df["actual_temperature_celsius"]
    )
    monitoring_df["absolute_error_celsius"] = monitoring_df["error_celsius"].abs()
    monitoring_df["squared_error_celsius"] = monitoring_df["error_celsius"] ** 2

    monitoring_df = monitoring_df.sort_values("predicted_event_time").reset_index(
        drop=True
    )

    monitoring_df["rolling_mae_7_predictions"] = (
        monitoring_df["absolute_error_celsius"].rolling(window=7, min_periods=1).mean()
    )

    return monitoring_df


def summarize_monitoring_results(monitoring_df: pd.DataFrame) -> dict:
    """Summarize monitoring errors."""
    y_true = monitoring_df["actual_temperature_celsius"]
    y_pred = monitoring_df["predicted_temperature_celsius"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    return {
        "n_matched_predictions": int(len(monitoring_df)),
        "monitoring_mae": float(mae),
        "monitoring_rmse": float(rmse),
        "latest_absolute_error_celsius": float(
            monitoring_df["absolute_error_celsius"].iloc[-1]
        ),
    }


def save_monitoring_results(monitoring_df: pd.DataFrame) -> None:
    """Save monitoring results locally."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    monitoring_df.to_csv(MONITORING_RESULTS_PATH, index=False)

    print()
    print("Saved monitoring results")
    print("------------------------")
    print(f"Monitoring results: {MONITORING_RESULTS_PATH}")


def main() -> None:
    """Run the complete monitoring workflow."""
    prediction_log_df = load_prediction_log()

    try:
        actual_temperature_df = fetch_actual_temperatures_for_predictions(
            prediction_log_df
        )

        monitoring_df = build_monitoring_results(
            prediction_log_df=prediction_log_df,
            actual_temperature_df=actual_temperature_df,
        )

    except ValueError as exc:
        print()
        print("Monitoring skipped")
        print("------------------")
        print(exc)
        return

    summary = summarize_monitoring_results(monitoring_df)

    print()
    print("Monitoring Summary")
    print("------------------")
    print(f"Matched predictions: {summary['n_matched_predictions']}")
    print(f"Monitoring MAE: {summary['monitoring_mae']:.4f}")
    print(f"Monitoring RMSE: {summary['monitoring_rmse']:.4f}")
    print(f"Latest absolute error: {summary['latest_absolute_error_celsius']:.4f} °C")

    save_monitoring_results(monitoring_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitor logged temperature predictions."
    )
    parser.parse_args()

    main()
