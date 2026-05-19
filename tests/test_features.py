import pandas as pd
import pytest

from config import TARGET_COLUMN
from features import (
    FEATURE_COLUMNS,
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_temperature_features,
    validate_raw_weather_dataframe,
)


def make_raw_weather_df(periods: int = 40) -> pd.DataFrame:
    event_time = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=periods,
        freq="h",
    )

    return pd.DataFrame(
        {
            "location_id": ["basel"] * periods,
            "event_time_unix": (event_time.view("int64") // 10**9).astype(int),
            "event_time": event_time,
            "temperature_2m": [float(i) for i in range(periods)],
            "relative_humidity_2m": [70 + (i % 10) for i in range(periods)],
            "precipitation": [0.1 * (i % 3) for i in range(periods)],
            "cloud_cover": [50 + (i % 20) for i in range(periods)],
            "pressure_msl": [1010.0 + 0.1 * i for i in range(periods)],
            "wind_speed_10m": [5.0 + 0.1 * i for i in range(periods)],
        }
    )


def test_validate_raw_weather_dataframe_rejects_empty_dataframe():
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="Weather dataframe is empty"):
        validate_raw_weather_dataframe(df)


def test_validate_raw_weather_dataframe_rejects_missing_columns():
    df = make_raw_weather_df().drop(columns=["temperature_2m"])

    with pytest.raises(ValueError, match="Missing required weather columns"):
        validate_raw_weather_dataframe(df)


def test_lag_features_are_computed_per_location():
    df = make_raw_weather_df(periods=30)

    result = add_lag_features(df)

    assert pd.isna(result.loc[0, "temperature_lag_1h"])
    assert result.loc[1, "temperature_lag_1h"] == 0.0
    assert result.loc[3, "temperature_lag_3h"] == 0.0
    assert result.loc[6, "temperature_lag_6h"] == 0.0
    assert result.loc[24, "temperature_lag_24h"] == 0.0


def test_rolling_features_do_not_include_current_row():
    df = make_raw_weather_df(periods=30)

    result = add_rolling_features(df)

    # At index 1, the rolling mean should only use index 0.
    assert result.loc[1, "temperature_mean_last_6h"] == 0.0

    # At index 6, the last 6 previous values are 0,1,2,3,4,5.
    assert result.loc[6, "temperature_mean_last_6h"] == pytest.approx(2.5)

    # Current value at index 6 is 6.0. If the current row were included,
    # the mean would be different. This protects against leakage.
    assert result.loc[6, "temperature_mean_last_6h"] != 6.0


def test_build_temperature_features_training_mode_adds_target():
    raw_df = make_raw_weather_df(periods=40)

    result = build_temperature_features(
        raw_df=raw_df,
        include_target=True,
    )

    assert TARGET_COLUMN in result.columns
    assert all(column in result.columns for column in FEATURE_COLUMNS)
    assert result[TARGET_COLUMN].isna().sum() == 0
    assert result[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_build_temperature_features_inference_mode_excludes_target():
    raw_df = make_raw_weather_df(periods=40)

    result = build_temperature_features(
        raw_df=raw_df,
        include_target=False,
    )

    assert TARGET_COLUMN not in result.columns
    assert all(column in result.columns for column in FEATURE_COLUMNS)
    assert result[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_build_temperature_features_sorts_by_event_time():
    raw_df = make_raw_weather_df(periods=40)
    shuffled_df = raw_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    result = build_temperature_features(
        raw_df=shuffled_df,
        include_target=True,
    )

    assert result["event_time"].is_monotonic_increasing


def test_add_calendar_features_adds_cyclical_features():
    raw_df = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 06:00:00",
                    "2026-01-01 12:00:00",
                    "2026-01-01 18:00:00",
                ]
            )
        }
    )

    result = add_calendar_features(raw_df)

    expected_columns = {
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    }

    assert expected_columns.issubset(result.columns)
    assert result["hour_sin"].between(-1, 1).all()
    assert result["hour_cos"].between(-1, 1).all()
    assert result["month_sin"].between(-1, 1).all()
    assert result["month_cos"].between(-1, 1).all()
