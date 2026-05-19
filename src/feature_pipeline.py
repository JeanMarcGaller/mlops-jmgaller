"""Feature pipeline for the temperature forecasting project.

This pipeline is responsible for creating the historical training dataset.

It performs the following steps:

1. Fetch historical hourly weather data from Open-Meteo.
2. Build model features and the supervised regression target.
3. Save a local Parquet feature cache for offline experiments and fallback use.
4. Optionally write the feature dataframe to the Hopsworks Training Feature Group.

The pipeline writes only historical rows with known target values. Live inference
rows are handled separately by `inference_pipeline.py` and stored in a separate
Inference Feature Group.
"""

import argparse
from datetime import date, timedelta

import pandas as pd

from config import (
    DATA_FEATURES_DIR,
    FEATURE_CACHE_PATH,
    FORECAST_HORIZON_HOURS,
    TARGET_COLUMN,
    TRAINING_FEATURE_GROUP_NAME,
    TRAINING_FEATURE_GROUP_VERSION,
)
from features import FEATURE_COLUMNS, build_temperature_features
from hopsworks_client import get_feature_store
from weather_api import fetch_historical_weather


def write_to_feature_store(features_df: pd.DataFrame) -> None:
    """Write engineered historical features to the Hopsworks Feature Store.

    The Feature Group is created if it does not exist yet. It uses:

    - `location_id` and `event_time_unix` as primary key
    - `event_time` as event-time column
    - `online_enabled=True` so the group can support online/inference access

    Args:
        features_df: Feature dataframe created by `build_temperature_features`
            with `include_target=True`.
    """
    fs = get_feature_store()

    feature_group = fs.get_or_create_feature_group(
        name=TRAINING_FEATURE_GROUP_NAME,
        version=TRAINING_FEATURE_GROUP_VERSION,
        description=(
            "Historical temperature forecasting features. "
            "Includes current weather features, lag features, rolling features, "
            "calendar features and regression target."
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
        f"Inserted {len(features_df)} rows into feature group "
        f"{TRAINING_FEATURE_GROUP_NAME}, version {TRAINING_FEATURE_GROUP_VERSION}."
    )


def build_and_cache_training_features(days: int) -> pd.DataFrame:
    """Fetch historical weather data and build cached training features.

    Historical data is fetched until two days before today. This buffer avoids
    requesting very recent historical data that may not yet be fully available
    from the archive API.

    Args:
        days: Number of historical days to fetch.

    Returns:
        Engineered training feature dataframe including the target column.
    """
    if days <= 0:
        raise ValueError("days must be greater than 0.")

    end_date = date.today() - timedelta(days=2)
    start_date = end_date - timedelta(days=days)

    print(f"Fetching historical weather data from {start_date} to {end_date}...")
    print(f"Forecast horizon: {FORECAST_HORIZON_HOURS} hours")
    print(f"Target column: {TARGET_COLUMN}")

    raw_df = fetch_historical_weather(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    print(f"Raw dataframe shape: {raw_df.shape}")

    features_df = build_temperature_features(
        raw_df=raw_df,
        include_target=True,
    )

    DATA_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(FEATURE_CACHE_PATH, index=False)

    print("Feature dataframe created successfully.")
    print(f"Shape: {features_df.shape}")
    print("Columns:")
    print(features_df.columns.tolist())
    print()
    print(features_df.head())
    print()
    print("Feature columns:")
    print(FEATURE_COLUMNS)
    print()
    print("Target summary:")
    print(features_df[TARGET_COLUMN].describe())

    print()
    print(f"Saved local feature cache to: {FEATURE_CACHE_PATH}")

    return features_df


def main(days: int, dry_run: bool) -> None:
    """Run the historical feature pipeline.

    Args:
        days: Number of historical days to fetch from Open-Meteo.
        dry_run: If True, build and cache features locally but skip the
            Hopsworks write step.
    """
    features_df = build_and_cache_training_features(days=days)

    if dry_run:
        print()
        print("Dry run only. Nothing was written to Hopsworks.")
        return

    write_to_feature_store(features_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build historical training features for temperature forecasting."
    )

    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of historical days to fetch from Open-Meteo.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create and validate the feature dataframe without writing to Hopsworks.",
    )

    args = parser.parse_args()

    main(
        days=args.days,
        dry_run=args.dry_run,
    )
