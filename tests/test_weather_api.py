from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from weather_api import (
    HOURLY_VARIABLES,
    fetch_forecast_weather,
    hourly_response_to_dataframe,
    validate_response,
)


def make_open_meteo_payload():
    return {
        "hourly": {
            "time": [
                "2026-01-01T00:00",
                "2026-01-01T01:00",
            ],
            "temperature_2m": [5.0, 4.5],
            "relative_humidity_2m": [80, 82],
            "precipitation": [0.0, 0.1],
            "cloud_cover": [50, 60],
            "pressure_msl": [1010.0, 1010.2],
            "wind_speed_10m": [5.0, 5.5],
        }
    }


def test_hourly_response_to_dataframe_returns_expected_schema():
    payload = make_open_meteo_payload()

    result = hourly_response_to_dataframe(payload)

    expected_columns = [
        "location_id",
        "event_time_unix",
        "event_time",
        *HOURLY_VARIABLES,
    ]

    assert result.columns.tolist() == expected_columns
    assert result.shape == (2, len(expected_columns))
    assert pd.api.types.is_datetime64_any_dtype(result["event_time"])
    assert result["event_time_unix"].notna().all()


def test_hourly_response_to_dataframe_rejects_empty_hourly_data():
    payload = {"hourly": {"time": []}}

    with pytest.raises(ValueError, match="empty hourly dataframe"):
        hourly_response_to_dataframe(payload)


def test_hourly_response_to_dataframe_rejects_missing_expected_variable():
    payload = make_open_meteo_payload()
    del payload["hourly"]["temperature_2m"]

    with pytest.raises(ValueError, match="missing expected columns"):
        hourly_response_to_dataframe(payload)


def test_validate_response_rejects_missing_hourly_section():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {}

    with pytest.raises(ValueError, match="hourly"):
        validate_response(response)


def test_validate_response_rejects_missing_hourly_time_values():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"hourly": {"temperature_2m": [1.0]}}

    with pytest.raises(ValueError, match="time"):
        validate_response(response)


def test_validate_response_wraps_http_error():
    response = Mock()
    response.status_code = 500
    response.text = "Internal Server Error"
    response.raise_for_status.side_effect = requests.HTTPError("server error")

    with pytest.raises(RuntimeError, match="Open-Meteo API request failed"):
        validate_response(response)


def test_validate_response_rejects_invalid_json():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.side_effect = ValueError("invalid json")

    with pytest.raises(RuntimeError, match="not valid JSON"):
        validate_response(response)


def test_fetch_forecast_weather_rejects_invalid_forecast_days():
    with pytest.raises(ValueError, match="forecast_days"):
        fetch_forecast_weather(forecast_days=0)
