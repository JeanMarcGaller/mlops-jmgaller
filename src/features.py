"""Feature engineering for the temperature forecasting project.

This module converts raw hourly weather observations/forecasts into a supervised
learning table for short-term temperature forecasting.

The central target is configured in `config.py`, for example:

    temperature_2m_next_6h

The feature engineering follows a common time-series-to-tabular approach:
current weather values, calendar features, lag features, rolling aggregates and
trend features are used to predict the temperature several hours ahead.

Important design decision:
Lag features, rolling features and the target are based on exact event_time
relationships instead of row positions. Rolling windows exclude the current row
with `closed="left"` to avoid data leakage.
"""

import numpy as np
import pandas as pd

from config import FORECAST_HORIZON_HOURS, TARGET_COLUMN

# These are the model input columns used consistently across training,
# tuning, feature selection, final training and inference.
FEATURE_COLUMNS = [
    # Current weather features.
    # These are known at the selected event_time and are therefore available
    # during both training and live inference.
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "pressure_msl",
    "wind_speed_10m",
    # Calendar features.
    # These help the model learn daily/weekly/seasonal temperature patterns.
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    # Lag features.
    # These represent past weather states relative to the current event_time.
    "temperature_lag_1h",
    "temperature_lag_3h",
    "temperature_lag_6h",
    "temperature_lag_18h",
    "temperature_lag_24h",
    "humidity_lag_1h",
    "pressure_lag_1h",
    "wind_speed_lag_1h",
    # Rolling aggregate features.
    # These summarize recent history using time-based windows that exclude
    # the current row.
    "temperature_mean_last_6h",
    "temperature_mean_last_24h",
    "humidity_mean_last_6h",
    "pressure_mean_last_6h",
    "precipitation_sum_last_24h",
    # Trend features.
    # These capture short-term movement rather than only absolute levels.
    "temperature_diff_1h",
    "pressure_diff_1h",
    "pressure_trend_last_6h",
]


RAW_WEATHER_COLUMNS = [
    "location_id",
    "event_time_unix",
    "event_time",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "pressure_msl",
    "wind_speed_10m",
]


def validate_raw_weather_dataframe(df: pd.DataFrame) -> None:
    """Validate that the raw weather dataframe has the required structure.

    The input dataframe is expected to come from `weather_api.py`. It must
    contain one row per location and event_time.

    Raises:
        ValueError: If the dataframe is empty or required columns are missing.
    """
    if df.empty:
        raise ValueError("Weather dataframe is empty.")

    missing_columns = [
        column for column in RAW_WEATHER_COLUMNS if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required weather columns: {missing_columns}")


def validate_unique_location_event_times(df: pd.DataFrame) -> None:
    """Validate that each location has at most one row per event_time."""
    duplicate_mask = df.duplicated(
        subset=["location_id", "event_time"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_rows = (
            df.loc[duplicate_mask, ["location_id", "event_time"]]
            .sort_values(["location_id", "event_time"])
            .head(10)
        )

        raise ValueError(
            "Duplicate rows found for location_id and event_time. "
            "Time-based feature joins require unique keys. "
            f"Examples: {duplicate_rows.to_dict(orient='records')}"
        )


def add_exact_hour_lookup(
    df: pd.DataFrame,
    source_column: str,
    output_column: str,
    offset_hours: int,
) -> pd.DataFrame:
    """Attach a value from an exact time offset.

    For each row at event_time t, this adds source_column from:

        t + offset_hours

    Examples:
        offset_hours=-24 attaches the value from exactly 24 hours before t.
        offset_hours=6 attaches the value from exactly 6 hours after t.
    """
    lookup_df = df[["location_id", "event_time", source_column]].copy()

    lookup_df["event_time"] = lookup_df["event_time"] - pd.Timedelta(hours=offset_hours)

    lookup_df = lookup_df.rename(
        columns={
            source_column: output_column,
        }
    )

    return df.merge(
        lookup_df,
        on=["location_id", "event_time"],
        how="left",
        validate="many_to_one",
    )


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features.

    Cyclical encodings represent periodic time patterns. They help the model
    understand that hour 23 is close to hour 0 and December is close to January.
    """
    df = df.copy()

    df["hour"] = df["event_time"].dt.hour
    df["day_of_week"] = df["event_time"].dt.dayofweek
    df["month"] = df["event_time"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add exact time-based lag features per location.

    Lag features use values from exact previous timestamps within the same
    location_id. If the exact timestamp is missing, the lag value remains null
    and the row is removed during missing-value cleanup.
    """
    df = df.copy()

    lag_specs = [
        ("temperature_2m", "temperature_lag_1h", -1),
        ("temperature_2m", "temperature_lag_3h", -3),
        ("temperature_2m", "temperature_lag_6h", -6),
        ("temperature_2m", "temperature_lag_18h", -18),
        ("temperature_2m", "temperature_lag_24h", -24),
        ("relative_humidity_2m", "humidity_lag_1h", -1),
        ("pressure_msl", "pressure_lag_1h", -1),
        ("wind_speed_10m", "wind_speed_lag_1h", -1),
    ]

    for source_column, output_column, offset_hours in lag_specs:
        df = add_exact_hour_lookup(
            df=df,
            source_column=source_column,
            output_column=output_column,
            offset_hours=offset_hours,
        )

    return df


def add_time_based_rolling_feature(
    df: pd.DataFrame,
    source_column: str,
    output_column: str,
    window: str,
    aggregation: str,
    min_periods: int = 1,
) -> pd.DataFrame:
    """Add a time-based rolling feature per location.

    The rolling window is based on event_time, not on row count. The current row
    is excluded with closed="left" to avoid using information from the prediction
    timestamp itself.

    Example:
        For event_time 10:00 and window="6h", the feature uses values from
        [04:00, 10:00), so 04:00 through 09:00 if hourly data is complete.
    """
    df = df.copy()
    result = pd.Series(index=df.index, dtype="float64")

    for _location_id, group_df in df.groupby("location_id", sort=False):
        group_df = group_df.sort_values("event_time")
        indexed_series = group_df.set_index("event_time")[source_column]

        rolling = indexed_series.rolling(
            window=window,
            closed="left",
            min_periods=min_periods,
        )

        if aggregation == "mean":
            values = rolling.mean()
        elif aggregation == "sum":
            values = rolling.sum()
        else:
            raise ValueError(f"Unsupported rolling aggregation: {aggregation}")

        result.loc[group_df.index] = values.to_numpy()

    df[output_column] = result

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based rolling aggregate features per location.

    Rolling windows are based on event_time instead of row counts. This makes
    the features robust to missing hourly rows. The current row is excluded from
    every rolling window to avoid leakage.
    """
    df = df.copy()

    rolling_specs = [
        ("temperature_2m", "temperature_mean_last_6h", "6h", "mean"),
        ("temperature_2m", "temperature_mean_last_24h", "24h", "mean"),
        ("relative_humidity_2m", "humidity_mean_last_6h", "6h", "mean"),
        ("pressure_msl", "pressure_mean_last_6h", "6h", "mean"),
        ("precipitation", "precipitation_sum_last_24h", "24h", "sum"),
    ]

    for source_column, output_column, window, aggregation in rolling_specs:
        df = add_time_based_rolling_feature(
            df=df,
            source_column=source_column,
            output_column=output_column,
            window=window,
            aggregation=aggregation,
            min_periods=1,
        )

    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add exact time-based trend features."""
    df = df.copy()

    df["temperature_diff_1h"] = df["temperature_2m"] - df["temperature_lag_1h"]
    df["pressure_diff_1h"] = df["pressure_msl"] - df["pressure_lag_1h"]

    df = add_exact_hour_lookup(
        df=df,
        source_column="pressure_msl",
        output_column="_pressure_lag_6h",
        offset_hours=-6,
    )

    df["pressure_trend_last_6h"] = df["pressure_msl"] - df["_pressure_lag_6h"]

    df = df.drop(columns=["_pressure_lag_6h"])

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the supervised learning target using an exact future timestamp.

    The target is the temperature exactly FORECAST_HORIZON_HOURS after the
    current event_time. If that future timestamp is missing, the target remains
    null and the row is removed during missing-value cleanup.
    """
    df = df.copy()

    df = add_exact_hour_lookup(
        df=df,
        source_column="temperature_2m",
        output_column=TARGET_COLUMN,
        offset_hours=FORECAST_HORIZON_HOURS,
    )

    return df


def summarize_missing_values(
    df: pd.DataFrame,
    columns: list[str],
) -> dict[str, int]:
    """Return missing-value counts for selected columns."""
    return {
        column: int(df[column].isna().sum())
        for column in columns
        if column in df.columns
    }


def drop_rows_with_missing_values(
    df: pd.DataFrame,
    include_target: bool,
) -> pd.DataFrame:
    """Drop rows with missing model inputs or target values."""
    required_columns = FEATURE_COLUMNS.copy()

    if include_target:
        required_columns.append(TARGET_COLUMN)

    missing_before = summarize_missing_values(df, required_columns)
    rows_before = len(df)

    df = df.dropna(subset=required_columns).copy()

    rows_after = len(df)
    dropped_rows = rows_before - rows_after

    if dropped_rows > 0:
        print()
        print("Missing-value cleanup")
        print("---------------------")
        print(f"Dropped rows: {dropped_rows}")
        print("Missing values before cleanup:")
        print({k: v for k, v in missing_before.items() if v > 0})

    return df


def build_temperature_features(
    raw_df: pd.DataFrame,
    include_target: bool,
) -> pd.DataFrame:
    """Build the final feature dataframe for training or inference.

    Args:
        raw_df: Raw hourly weather dataframe from `weather_api.py`.
        include_target: If True, add the future temperature target. Use True
            for historical training data and False for live inference data.

    Returns:
        A dataframe with primary keys, event_time, model features and optionally
        the target column.
    """
    validate_raw_weather_dataframe(raw_df)

    df = raw_df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"])

    # Sorting keeps feature generation deterministic before time-based lag,
    # rolling and target calculations.
    df = df.sort_values(["location_id", "event_time"]).reset_index(drop=True)

    validate_unique_location_event_times(df)

    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_trend_features(df)

    if include_target:
        df = add_target(df)

    df = drop_rows_with_missing_values(
        df=df,
        include_target=include_target,
    )

    output_columns = [
        "location_id",
        "event_time_unix",
        "event_time",
        *FEATURE_COLUMNS,
    ]

    if include_target:
        output_columns.append(TARGET_COLUMN)

    return df[output_columns].reset_index(drop=True)
