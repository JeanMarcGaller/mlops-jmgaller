# src/features.py

## Role in the Project

`features.py` contains the feature engineering logic of the project. It converts raw hourly weather data from `weather_api.py` into a tabular dataset for short-term temperature forecasting.

The module is used in two modes:

- training mode with `include_target=True`
- inference mode with `include_target=False`

In training mode, the module adds the future temperature target. In inference mode, it creates only the model input features because the future actual temperature is not known yet.

## Inputs

The input is the raw weather dataframe produced by `weather_api.py`.

Expected columns:

```text
location_id
event_time_unix
event_time
temperature_2m
relative_humidity_2m
precipitation
cloud_cover
pressure_msl
wind_speed_10m
```

The module also reads forecasting configuration from `config.py`:

* `FORECAST_HORIZON_HOURS`
* `TARGET_COLUMN`

## Outputs

The output is a feature dataframe containing:

* primary key columns
* event-time column
* model feature columns
* optionally the target column

Training output contains:

```text
location_id
event_time_unix
event_time
FEATURE_COLUMNS
TARGET_COLUMN
```

Inference output contains:

```text
location_id
event_time_unix
event_time
FEATURE_COLUMNS
```

## Key Constants

### `FEATURE_COLUMNS`

`FEATURE_COLUMNS` defines the shared model input schema used across the project:

* baseline training
* tuning
* feature selection
* final training
* model comparison
* backtesting
* inference

The features are grouped into:

* current weather features
* calendar features
* cyclic calendar features
* lag features
* rolling aggregate features
* trend features

### `RAW_WEATHER_COLUMNS`

`RAW_WEATHER_COLUMNS` defines the expected input schema from `weather_api.py`.

This makes the interface between data ingestion and feature engineering explicit.

## Key Functions

### `validate_raw_weather_dataframe()`

Validates that the raw weather dataframe is not empty and contains all required raw weather columns.

### `validate_unique_location_event_times()`

Validates that each `location_id` and `event_time` combination occurs at most once.

This is important because time-based lag and target joins require unique keys. If duplicate timestamps existed for the same location, an exact timestamp lookup could become ambiguous.

### `add_exact_hour_lookup()`

Adds a value from an exact timestamp offset.

For each row at event time `t`, it attaches the selected source value from:

```text
t + offset_hours
```

Examples:

```text
offset_hours = -24  -> value from exactly 24 hours before t
offset_hours = 6    -> value from exactly 6 hours after t
```

This function is the basis for both lag features and target generation.

### `add_calendar_features()`

Adds calendar-based features:

```text
hour
day_of_week
month
is_weekend
hour_sin
hour_cos
month_sin
month_cos
```

The sine and cosine encodings represent cyclic time patterns. This helps the model understand that hour 23 is close to hour 0 and that December is close to January.

### `add_lag_features()`

Creates exact timestamp-based lag features per location.

Examples:

```text
temperature_lag_1h
temperature_lag_6h
temperature_lag_24h
humidity_lag_1h
pressure_lag_1h
wind_speed_lag_1h
```

If the exact previous timestamp is missing, the lag value remains missing and the row is removed later during missing-value cleanup.

### `add_time_based_rolling_feature()`

Creates one time-based rolling aggregate feature per location.

The rolling window is based on `event_time`.

For example, for `event_time = 10:00` and `window = "6h"`, the feature uses:

```text
[04:00, 10:00)
```

The current row is excluded through `closed="left"`. This prevents information from the prediction timestamp from leaking into its own historical aggregate.

### `add_rolling_features()`

Creates all rolling aggregate features:

```text
temperature_mean_last_6h
temperature_mean_last_24h
humidity_mean_last_6h
pressure_mean_last_6h
precipitation_sum_last_24h
```

Mean aggregation is used for temperature, humidity and pressure. Sum aggregation is used for precipitation.

### `add_trend_features()`

Creates short-term trend features:

```text
temperature_diff_1h
pressure_diff_1h
pressure_trend_last_6h
```

These features describe recent movement instead of only absolute levels.

### `add_target()`

Creates the supervised learning target using an exact future timestamp.

For a six-hour forecast horizon, the target for:

```text
event_time = 2026-05-01 12:00:00
```

is:

```text
temperature_2m at 2026-05-01 18:00:00
```

If the exact future timestamp is missing, the target remains missing and the row is removed during missing-value cleanup.

### `summarize_missing_values()`

Counts missing values for selected columns.

This is used to make missing-value cleanup transparent.

### `drop_rows_with_missing_values()`

Drops rows with missing required model inputs.

In training mode, it also drops rows with a missing target.

The function prints how many rows were removed and which feature or target columns had missing values before cleanup.

### `build_temperature_features()`

Main orchestration function.

It performs the full feature engineering process:

1. validate raw input schema
2. convert `event_time` to datetime
3. sort by `location_id` and `event_time`
4. validate unique location/timestamp pairs
5. add calendar features
6. add exact timestamp-based lag features
7. add time-based rolling features
8. add trend features
9. optionally add the target
10. remove incomplete rows
11. return the final dataframe in a stable column order

## Key Design Decisions

### Shared Training and Inference Logic

The same feature engineering function is used for historical training data and live inference data.

The only difference is whether the future target is added.

### Exact Timestamp-Based Lag Features

Lag features are created through exact `location_id` and `event_time` joins.

This avoids the problem of row-based shifts. If an hourly row is missing, the pipeline does not silently use the wrong neighboring row.

### Exact Timestamp-Based Target

The target is generated from the exact future timestamp defined by `FORECAST_HORIZON_HOURS`.

This ensures that `temperature_2m_next_6h` really means the temperature exactly six hours after the feature timestamp.

### Time-Based Rolling Windows

Rolling features use real time windows such as `6h` or `24h`.

This makes them more robust to missing hourly rows than fixed row-count rolling windows.

### Leakage Prevention

Rolling windows exclude the current row with `closed="left"`.

This prevents the current prediction timestamp from being included in its own historical aggregate.

### Explicit Missing-Value Cleanup

Missing values from lag, rolling or target creation are not silently imputed.

Rows with missing required model features or targets are removed and the missing-value counts are printed.

## Data Leakage / Point-in-Time Considerations

The module is designed to reduce time-series leakage:

* lag features use only exact previous timestamps
* rolling features exclude the current timestamp
* the target is only created for historical training data
* inference mode does not create a target
* missing timestamps produce missing values instead of e.g. incorrect row-based matches

## Validation and Error Handling

The module validates:

* non-empty raw weather dataframe
* required raw weather columns
* unique `location_id` and `event_time` pairs
* supported rolling aggregation methods

Rows with missing required features or targets are removed after feature creation.

## Limitations / Future Work

* Raw weather value ranges are not fully validated yet.
* Time gaps are handled safely through timestamp joins, but not separately reported.
* Event-time handling could be hardened with an explicit UTC-normalized timestamp.
* A production-grade version could add stronger data quality checks for missing raw values, physical value ranges and time-series gaps.
