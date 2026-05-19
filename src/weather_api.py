"""Open-Meteo API client for the temperature forecasting project.

This module retrieves raw hourly weather data from Open-Meteo and converts the
response into a dataframe used by the feature pipeline.

Two API endpoints are used:

1. Historical Weather API
   Used by the feature pipeline to create training data with known future
   target values.

2. Forecast API
   Used by the inference pipeline to create live inference features.

The returned dataframe always follows the same schema:

    location_id
    event_time_unix
    event_time
    temperature_2m
    relative_humidity_2m
    precipitation
    cloud_cover
    pressure_msl
    wind_speed_10m

The `event_time` value comes directly from Open-Meteo and is used as the
event-time column in hopsworks. `event_time_unix` is used as part of the
hopsworks primary key together with `location_id`.
"""

from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests

from config import LATITUDE, LOCATION_ID, LONGITUDE, TIMEZONE

# Open-Meteo endpoint URLs.
# Historical data is used for training; forecast data is used for inference.
HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Timeout for external API calls in seconds.
REQUEST_TIMEOUT_SECONDS = 30

# Hourly variables requested from Open-Meteo.
# Keep this list aligned with the raw weather columns expected by `features.py`.
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "pressure_msl",
    "wind_speed_10m",
]


def validate_response(response: requests.Response) -> dict[str, Any]:
    """Validate an Open-Meteo response and return the parsed JSON payload.

    Args:
        response: HTTP response returned by `requests.get`.

    Returns:
        Parsed Open-Meteo JSON response.

    Raises:
        RuntimeError: If the HTTP request failed or the response body is not
            valid JSON.
        ValueError: If the response does not contain the expected hourly data.
    """
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            "Open-Meteo API request failed with status "
            f"{response.status_code}: {response.text}"
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Open-Meteo API response is not valid JSON.") from exc

    if "hourly" not in data:
        raise ValueError("Open-Meteo response does not contain an 'hourly' section.")

    if "time" not in data["hourly"]:
        raise ValueError("Open-Meteo response does not contain hourly 'time' values.")

    return data


def hourly_response_to_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    """Convert Open-Meteo hourly JSON data into the project dataframe schema.

    Args:
        data: Parsed Open-Meteo JSON response containing an `hourly` section.

    Returns:
        Hourly weather dataframe with canonical column order.

    Raises:
        ValueError: If the converted dataframe is empty or expected variables
            are missing.
    """
    hourly = data["hourly"]

    df = pd.DataFrame(hourly)

    if df.empty:
        raise ValueError("Open-Meteo returned an empty hourly dataframe.")

    df = df.rename(columns={"time": "event_time"})

    # Open-Meteo returns local timestamps when a timezone is passed in the API
    # parameters. The project keeps them timezone-naive for consistent downstream
    # feature engineering, logging and Hopsworks writes.
    df["event_time"] = pd.to_datetime(df["event_time"])
    df["location_id"] = LOCATION_ID

    # Stable numeric timestamp used together with location_id as Hopsworks
    # primary key. This avoids using a datetime object directly as key.
    df["event_time_unix"] = df["event_time"].astype("int64") // 10**9

    ordered_columns = [
        "location_id",
        "event_time_unix",
        "event_time",
        *HOURLY_VARIABLES,
    ]

    missing_columns = [column for column in ordered_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Open-Meteo response is missing expected columns: {missing_columns}"
        )

    return df[ordered_columns].copy()


def fetch_historical_weather(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch historical hourly weather data from Open-Meteo.

    This function is used by the feature pipeline. Historical data is required
    because the future target value is known for past timestamps.

    Args:
        start_date: Start date in ISO format, for example "2025-04-26".
        end_date: End date in ISO format, for example "2026-04-26".

    Returns:
        Canonical raw weather dataframe.
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": TIMEZONE,
    }

    response = requests.get(
        HISTORICAL_WEATHER_URL,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    data = validate_response(response)

    return hourly_response_to_dataframe(data)


def fetch_forecast_weather(
    forecast_days: int = 2,
    past_days: int = 0,
) -> pd.DataFrame:
    """Fetch current forecast weather data from Open-Meteo.

    This function is used by the inference pipeline. The returned rows can include
    recent past forecast context and future forecast rows.

    Args:
        forecast_days: Number of forecast days requested from Open-Meteo.
        past_days: Number of recent past days requested from Open-Meteo.

    Returns:
        Canonical raw weather dataframe.

    Raises:
        ValueError: If forecast_days is smaller than 1 or past_days is negative.
    """
    if forecast_days < 1:
        raise ValueError("forecast_days must be greater than or equal to 1.")

    if past_days < 0:
        raise ValueError("past_days must be greater than or equal to 0.")

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": TIMEZONE,
        "forecast_days": forecast_days,
        "past_days": past_days,
    }

    response = requests.get(
        FORECAST_WEATHER_URL,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    data = validate_response(response)

    return hourly_response_to_dataframe(data)


def main() -> None:
    """Run a lightweight manual smoke test for the Open-Meteo API client."""
    end_date = date.today() - timedelta(days=2)
    start_date = end_date - timedelta(days=7)

    print(f"Fetching historical weather from {start_date} to {end_date}...")

    historical_df = fetch_historical_weather(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    print("Historical dataframe:")
    print(historical_df.shape)
    print(historical_df.head())
    print(historical_df.tail())

    print()
    print("Fetching forecast weather...")

    forecast_df = fetch_forecast_weather(forecast_days=2)

    print("Forecast dataframe:")
    print(forecast_df.shape)
    print(forecast_df.head())
    print(forecast_df.tail())


if __name__ == "__main__":
    main()
