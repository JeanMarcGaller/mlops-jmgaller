"""Inference pipeline for the temperature forecasting project.

This pipeline runs a live temperature forecast using the latest model from the
Hopsworks Model Registry.

It performs the following steps:

1. Load the configured or latest model version from Hopsworks Model Registry.
2. Fetch current Open-Meteo forecast data.
3. Build live inference features without target column.
4. Select the next available forecast hour.
5. Write the inference feature row to the Hopsworks Inference Feature Group.
6. Retrieve the same feature vector via the Hopsworks Inference Feature View.
7. Prepare model input using the exact feature order stored in the model bundle.
8. Predict the temperature for the configured forecast horizon.

Important design decision:
The model input columns are loaded from the model bundle. This ensures inference
uses the exact same feature schema and order that were used during training.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from config import (
    FORECAST_HORIZON_HOURS,
    INFERENCE_FEATURE_GROUP_NAME,
    INFERENCE_FEATURE_GROUP_VERSION,
    INFERENCE_FEATURE_VIEW_NAME,
    INFERENCE_FEATURE_VIEW_VERSION,
    LATEST_PREDICTION_PATH,
    MODEL_NAME,
    MODEL_VERSION,
    PREDICTION_LOG_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
    TIMEZONE,
)
from features import build_temperature_features
from hopsworks_client import get_feature_store, get_model_registry
from weather_api import fetch_forecast_weather

INFERENCE_PAST_DAYS = 2
INFERENCE_FORECAST_DAYS = 2


def get_latest_model_version(model_registry, model_name: str) -> int:
    """Return the latest available model version from Hopsworks Model Registry.

    Args:
        model_registry: Hopsworks Model Registry object.
        model_name: Registered model name.

    Returns:
        Highest model version number.

    Raises:
        ValueError: If no model versions exist for the given model name.
    """
    models = model_registry.get_models(name=model_name)

    if not models:
        raise ValueError(f"No models found in registry for name: {model_name}")

    versions = [model.version for model in models]
    latest_version = max(versions)

    return latest_version


def load_model_from_registry() -> tuple[dict, int]:
    """Load a model bundle from the Hopsworks Model Registry.

    If MODEL_VERSION is set in `.env`, this exact model version is loaded.
    If MODEL_VERSION is empty, the latest available version is loaded.

    Returns:
        Tuple of model bundle loaded from `model.joblib` and resolved model version.

    Raises:
        FileNotFoundError: If the downloaded registry artifact does not contain
            `model.joblib`.
    """
    print("Loading model from Hopsworks Model Registry...")

    model_registry = get_model_registry()

    if MODEL_VERSION is None:
        resolved_model_version = get_latest_model_version(
            model_registry=model_registry,
            model_name=MODEL_NAME,
        )

        print(
            "Loading latest model version for: "
            f"{MODEL_NAME}, version {resolved_model_version}"
        )
    else:
        resolved_model_version = MODEL_VERSION

        print(
            "Loading configured model version: "
            f"{MODEL_NAME}, version {resolved_model_version}"
        )

    model_meta = model_registry.get_model(
        name=MODEL_NAME,
        version=resolved_model_version,
    )

    download_dir = model_meta.download()
    model_path = Path(download_dir) / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found in registry download: {model_path}"
        )

    model_bundle = joblib.load(model_path)

    print(f"Loaded model from registry: {MODEL_NAME}, version {resolved_model_version}")

    return model_bundle, resolved_model_version


def fetch_live_weather_context(
    past_days: int = INFERENCE_PAST_DAYS,
    forecast_days: int = INFERENCE_FORECAST_DAYS,
) -> pd.DataFrame:
    """Fetch recent past weather context and future forecast rows for inference.

    The past context is required so lag, rolling and trend features are available
    for the next forecast hour.
    """
    print(
        "Fetching live weather context from Open-Meteo "
        f"with past_days={past_days}, forecast_days={forecast_days}..."
    )

    raw_df = fetch_forecast_weather(
        past_days=past_days,
        forecast_days=forecast_days,
    )

    raw_df = raw_df.copy()
    raw_df["event_time"] = pd.to_datetime(raw_df["event_time"])

    raw_df = (
        raw_df.drop_duplicates(
            subset=["location_id", "event_time"],
            keep="last",
        )
        .sort_values(["location_id", "event_time"])
        .reset_index(drop=True)
    )

    print(f"Live weather context shape: {raw_df.shape}")
    print(f"context_from: {raw_df['event_time'].min()}")
    print(f"context_until: {raw_df['event_time'].max()}")

    return raw_df


def build_current_inference_row() -> pd.DataFrame:
    """Build the latest live inference row from Open-Meteo forecast data.

    The function fetches forecast data, applies the same feature engineering as
    training without adding a target, and selects the next full forecast hour in
    the configured local timezone.

    Returns:
        One-row dataframe containing the selected inference feature row.

    Raises:
        ValueError: If no suitable current or future forecast row is available
            after feature engineering.
    """

    raw_df = fetch_live_weather_context()

    features_df = build_temperature_features(
        raw_df=raw_df,
        include_target=False,
    )

    # Use the next full forecast hour in the configured local timezone.
    # Example:
    #   current time: 10:23
    #   selected hour: 11:00
    #
    # Open-Meteo returns local timestamps because TIMEZONE is sent in the API
    # request. We convert the current timestamp to timezone-naive to match the
    # event_time format in the feature dataframe.
    next_forecast_hour = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None).ceil("h")

    future_rows = features_df[features_df["event_time"] >= next_forecast_hour].copy()

    if future_rows.empty:
        raise ValueError(
            "No current or future forecast rows available from Open-Meteo "
            "after feature engineering. Try increasing forecast_days."
        )

    inference_row = future_rows.sort_values("event_time").head(1).copy()

    selected_event_time = pd.Timestamp(inference_row["event_time"].iloc[0])

    if selected_event_time > next_forecast_hour:
        raise ValueError(
            "The first usable inference row is later than the next forecast hour. "
            f"next_forecast_hour={next_forecast_hour}, "
            f"selected_event_time={selected_event_time}. "
            "Increase INFERENCE_PAST_DAYS or inspect missing feature values."
        )

    print(f"Selected forecast event_time: {inference_row['event_time'].iloc[0]}")
    print(f"Prediction target: {TARGET_COLUMN}")
    print(
        f"This means: predict temperature {FORECAST_HORIZON_HOURS} hours "
        "after the selected event_time."
    )
    print()
    print("Current inference row:")
    print(inference_row)

    return inference_row


def write_inference_features_to_hopsworks(features_df: pd.DataFrame) -> None:
    """Write the latest inference feature row to Hopsworks.

    The inference Feature Group is separate from the training Feature Group
    because live inference rows do not have a known future target yet.

    Args:
        features_df: Inference feature dataframe, usually containing one row.
    """
    print("Writing latest inference features to Hopsworks...")

    fs = get_feature_store()

    feature_group = fs.get_or_create_feature_group(
        name=INFERENCE_FEATURE_GROUP_NAME,
        version=INFERENCE_FEATURE_GROUP_VERSION,
        description=(
            "Live temperature forecasting features. "
            "Contains Open-Meteo forecast-based features without target."
        ),
        primary_key=["location_id", "event_time_unix"],
        event_time="event_time",
        online_enabled=True,
    )

    feature_group.insert(
        features_df,
        wait=True,
    )

    print(
        f"Inserted {len(features_df)} inference feature row(s) into "
        f"{INFERENCE_FEATURE_GROUP_NAME}, version {INFERENCE_FEATURE_GROUP_VERSION}."
    )


def get_or_create_inference_feature_view(fs):
    """Create or load the Hopsworks Inference Feature View.

    The Feature View provides the serving-time access path for retrieving the
    feature vector used by the model.

    Args:
        fs: Hopsworks Feature Store object.

    Returns:
        Hopsworks Feature View object.
    """
    feature_group = fs.get_feature_group(
        name=INFERENCE_FEATURE_GROUP_NAME,
        version=INFERENCE_FEATURE_GROUP_VERSION,
    )

    query = feature_group.select_all()

    try:
        feature_view = fs.get_feature_view(
            name=INFERENCE_FEATURE_VIEW_NAME,
            version=INFERENCE_FEATURE_VIEW_VERSION,
        )

        if feature_view is not None:
            print(
                "Using existing inference feature view: "
                f"{INFERENCE_FEATURE_VIEW_NAME}, "
                f"version {INFERENCE_FEATURE_VIEW_VERSION}"
            )
            return feature_view

    except Exception as exc:
        print(
            f"Inference feature view {INFERENCE_FEATURE_VIEW_NAME}, "
            f"version {INFERENCE_FEATURE_VIEW_VERSION} was not found."
        )
        print(f"Reason: {exc}")

    print(
        f"Creating inference feature view: "
        f"{INFERENCE_FEATURE_VIEW_NAME}, version {INFERENCE_FEATURE_VIEW_VERSION}"
    )

    feature_view = fs.create_feature_view(
        name=INFERENCE_FEATURE_VIEW_NAME,
        version=INFERENCE_FEATURE_VIEW_VERSION,
        query=query,
        description=(
            "Inference feature view for live temperature forecasting features."
        ),
    )

    if feature_view is None:
        raise RuntimeError(
            "Inference feature view creation returned None. "
            "Increase INFERENCE_FEATURE_VIEW_VERSION or check Hopsworks."
        )

    return feature_view


def has_unnamed_columns(df: pd.DataFrame) -> bool:
    """Return whether a dataframe appears to have unnamed positional columns."""
    return all(isinstance(column, int) for column in df.columns)


def normalize_feature_vector(
    feature_vector,
    expected_columns: list[str],
) -> pd.DataFrame:
    """Convert a Hopsworks feature vector response into a dataframe.

    Depending on Hopsworks client behavior, `get_feature_vector` can return a
    dataframe, dictionary or sequence-like object. This helper normalizes the
    response and restores column names only when the response has unnamed
    positional columns.

    Args:
        feature_vector: Raw feature vector returned by Hopsworks.
        expected_columns: Expected column order based on the inserted
            inference row.

    Returns:
        Feature vector as a dataframe.

    Raises:
        ValueError: If the response schema cannot be safely matched to the
            expected feature schema.
    """
    if isinstance(feature_vector, pd.DataFrame):
        feature_df = feature_vector.copy()
    elif isinstance(feature_vector, dict):
        feature_df = pd.DataFrame([feature_vector])
    else:
        feature_df = pd.DataFrame([feature_vector])

    expected_column_set = set(expected_columns)

    if expected_column_set.issubset(feature_df.columns):
        return feature_df

    if has_unnamed_columns(feature_df):
        if feature_df.shape[1] != len(expected_columns):
            raise ValueError(
                "Unnamed feature vector length does not match expected schema. "
                f"Expected {len(expected_columns)} columns, "
                f"got {feature_df.shape[1]}."
            )

        feature_df.columns = expected_columns

        return feature_df

    missing_columns = sorted(expected_column_set - set(feature_df.columns))

    raise ValueError(
        "Feature vector dataframe does not contain the expected columns and "
        "does not use unnamed positional columns. Refusing to rename columns "
        "because that could silently misalign feature values. "
        f"Missing columns: {missing_columns}. "
        f"Received columns: {feature_df.columns.tolist()}"
    )


def load_feature_vector_from_feature_view(
    location_id: str,
    event_time_unix: int,
    expected_columns: list[str],
) -> pd.DataFrame:
    """Load one inference feature vector from the Hopsworks Feature View.

    Args:
        location_id: Location identifier, for example "basel".
        event_time_unix: Unix timestamp used as part of the primary key.
        expected_columns: Expected column names of the feature vector.

    Returns:
        Feature vector dataframe.
    """
    print("Loading inference feature vector from Hopsworks Feature View...")

    fs = get_feature_store()

    feature_view = get_or_create_inference_feature_view(fs)

    feature_vector = feature_view.get_feature_vector(
        entry={
            "location_id": location_id,
            "event_time_unix": event_time_unix,
        }
    )

    feature_df = normalize_feature_vector(
        feature_vector=feature_vector,
        expected_columns=expected_columns,
    )

    print("Loaded feature vector from Feature View.")
    print(feature_df)

    return feature_df


def prepare_model_input(
    feature_vector_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Prepare model input with the exact feature order from training.

    Args:
        feature_vector_df: Feature vector dataframe loaded from Hopsworks.
        feature_columns: Feature columns stored in the model bundle.

    Returns:
        Dataframe containing only model input columns in the correct order.

    Raises:
        ValueError: If one or more required model input columns are missing.
    """
    missing_columns = [
        column for column in feature_columns if column not in feature_vector_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing model input columns from feature vector: {missing_columns}"
        )

    X = feature_vector_df[feature_columns].copy()

    return X


def predict_temperature(
    model,
    X: pd.DataFrame,
) -> float:
    """Run model prediction and return the predicted temperature as float."""
    prediction = model.predict(X)

    return float(prediction[0])


def print_inference_result(
    location_id: str,
    event_time,
    predicted_temperature: float,
    metrics: dict,
) -> None:
    """Print a human-readable inference result summary."""
    predicted_event_time = pd.Timestamp(event_time) + pd.Timedelta(
        hours=FORECAST_HORIZON_HOURS
    )

    print()
    print("Inference Result")
    print("----------------")
    print(f"location_id: {location_id}")
    print(f"feature_event_time: {event_time}")
    print(f"predicted_event_time: {predicted_event_time}")
    print(f"target: {TARGET_COLUMN}")
    print(f"predicted_temperature_celsius: {predicted_temperature:.2f}")

    if "test_mae" in metrics:
        print(f"model_test_mae: {metrics['test_mae']:.4f}")

    if "test_rmse" in metrics:
        print(f"model_test_rmse: {metrics['test_rmse']:.4f}")

    if "mae_improvement_over_best_naive_baseline" in metrics:
        print(
            "mae_improvement_over_best_naive_baseline: "
            f"{metrics['mae_improvement_over_best_naive_baseline']:.2%}"
        )
    elif "mae_improvement_over_baseline" in metrics:
        print(
            "mae_improvement_over_baseline: "
            f"{metrics['mae_improvement_over_baseline']:.2%}"
        )


def validate_model_bundle(model_bundle: dict) -> None:
    """Validate that the loaded model bundle contains required inference data.

    Raises:
        ValueError: If required model bundle keys are missing.
    """
    required_keys = ["model", "feature_columns"]

    missing_keys = [key for key in required_keys if key not in model_bundle]

    if missing_keys:
        raise ValueError(f"Model bundle is missing required keys: {missing_keys}")


def build_prediction_record(
    location_id: str,
    feature_event_time,
    predicted_event_time,
    target_column: str,
    predicted_temperature: float,
    model_metrics: dict,
    model_version,
) -> dict:
    """Build a JSON-serializable prediction record."""
    return {
        "location_id": location_id,
        "feature_event_time": str(feature_event_time),
        "predicted_event_time": str(predicted_event_time),
        "target_column": target_column,
        "predicted_temperature_celsius": float(predicted_temperature),
        "model_version": None if model_version is None else str(model_version),
        "model_test_mae": model_metrics.get("test_mae"),
        "model_test_rmse": model_metrics.get("test_rmse"),
        "mae_improvement_over_best_naive_baseline": model_metrics.get(
            "mae_improvement_over_best_naive_baseline"
        ),
    }


def save_latest_prediction(prediction_record: dict) -> None:
    """Save latest prediction as JSON."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with LATEST_PREDICTION_PATH.open("w", encoding="utf-8") as file:
        json.dump(prediction_record, file, indent=2)

    print()
    print(f"Saved latest prediction to: {LATEST_PREDICTION_PATH}")


def append_prediction_log(prediction_record: dict) -> None:
    """Append prediction record to local CSV prediction log."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    row_df = pd.DataFrame([prediction_record])

    if PREDICTION_LOG_PATH.exists():
        existing_df = pd.read_csv(PREDICTION_LOG_PATH)
        log_df = pd.concat([existing_df, row_df], axis=0, ignore_index=True)
    else:
        log_df = row_df

    log_df.to_csv(PREDICTION_LOG_PATH, index=False)

    print(f"Appended prediction to: {PREDICTION_LOG_PATH}")


def save_prediction_artifacts(prediction_record: dict) -> None:
    """Save latest prediction and append prediction log."""
    save_latest_prediction(prediction_record)
    append_prediction_log(prediction_record)


def main() -> None:
    """Run the complete live inference workflow."""
    model_bundle, resolved_model_version = load_model_from_registry()
    validate_model_bundle(model_bundle)

    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    metrics = model_bundle.get("metrics", {})

    inference_row = build_current_inference_row()

    write_inference_features_to_hopsworks(inference_row)

    location_id = inference_row["location_id"].iloc[0]
    event_time = inference_row["event_time"].iloc[0]
    event_time_unix = int(inference_row["event_time_unix"].iloc[0])

    expected_columns = inference_row.columns.tolist()

    feature_vector_df = load_feature_vector_from_feature_view(
        location_id=location_id,
        event_time_unix=event_time_unix,
        expected_columns=expected_columns,
    )

    X = prepare_model_input(
        feature_vector_df=feature_vector_df,
        feature_columns=feature_columns,
    )

    predicted_temperature = predict_temperature(
        model=model,
        X=X,
    )

    predicted_event_time = pd.to_datetime(event_time) + pd.Timedelta(
        hours=FORECAST_HORIZON_HOURS
    )

    prediction_record = build_prediction_record(
        location_id=str(location_id),
        feature_event_time=event_time,
        predicted_event_time=predicted_event_time,
        target_column=TARGET_COLUMN,
        predicted_temperature=predicted_temperature,
        model_metrics=metrics,
        model_version=resolved_model_version,
    )

    save_prediction_artifacts(prediction_record)

    print_inference_result(
        location_id=location_id,
        event_time=event_time,
        predicted_temperature=predicted_temperature,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
