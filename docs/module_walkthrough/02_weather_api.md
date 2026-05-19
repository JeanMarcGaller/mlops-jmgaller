# src/weather_api.py

## Role in the Project

`weather_api.py` is the data source adapter for Open-Meteo. It retrieves raw hourly weather data and converts Open-Meteo responses into a dataframe schema used by the rest of the project.

The module is used by:

- the feature pipeline to load historical training data
- the inference pipeline to load live forecast data
- the monitoring pipeline to load historical actual temperatures

## Inputs

The module reads location settings from `config.py`:

- `LATITUDE`
- `LONGITUDE`
- `LOCATION_ID`
- `TIMEZONE`

It uses two Open-Meteo endpoints:

- Historical Weather API
- Forecast API

The historical endpoint is used when the later target value is already known. The forecast endpoint is used when creating live inference features.

## Outputs

Both historical and forecast API responses are converted into the same dataframe schema:

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

## Key Functions

### `validate_response()`

Validates the raw HTTP response from Open-Meteo.

It checks:

* whether the HTTP request was successful
* whether the response body is valid JSON
* whether the response contains an `hourly` section
* whether the `hourly` section contains `time` values

### `hourly_response_to_dataframe()`

Converts Open-Meteo hourly JSON data into the project dataframe schema:

* converts Open-Meteo `time` values into `event_time`
* parses `event_time` as pandas datetime
* adds the configured `location_id`
* creates `event_time_unix`
* validates that all expected weather columns are present
* returns columns in a stable order

### `fetch_historical_weather()`

Fetches historical hourly weather data for a given date range.

This function is used for training data generation because historical rows have known future target values. It is also used by monitoring to retrieve actual temperatures for prediction timestamps that are already in the past.

### `fetch_forecast_weather()`

Fetches current forecast weather data.

This function is used by the inference pipeline. It supports:

* `forecast_days`
* `past_days`

`past_days` is important for live inference because lag, rolling and trend features require recent past context. Without recent past rows, the first forecast hours would have missing lag and rolling values and could not be used as immediate inference rows.

## Key Design Decisions

### Same Schema for Historical and Forecast Data

Historical weather data and forecast data are normalized into the same dataframe structure. This keeps feature engineering independent of the original Open-Meteo endpoint.

### Location ID Added During Ingestion

`location_id` is added immediately after loading the data. The current project uses Basel only, but the schema is compatible with multiple locations.

### Numeric Event-Time Key

`event_time_unix` is derived from `event_time` and used together with `location_id` as the hopsworks primary key.

This avoids using a datetime object directly as the key and gives each location and timestamp an identifier.

### Forecast Context for Live Inference

The Forecast API call can include recent past rows through `past_days`.

This is required because the inference pipeline predicts the next forecast hour, but features such as `temperature_lag_24h`, `temperature_mean_last_24h` and `pressure_trend_last_6h` need historical context before that selected forecast timestamp.

## Validation and Error Handling

The module raises errors for:

* failed HTTP requests
* invalid JSON responses
* missing `hourly` data
* missing hourly `time` values
* empty hourly dataframes
* missing expected weather columns
* invalid forecast parameters such as `forecast_days < 1` or `past_days < 0`

This makes external API and schema problems visible before data reaches feature engineering, hopsworks or model training.

## Point-in-Time / Leakage Considerations

`weather_api.py` itself does not create labels, lag features or rolling features. Therefore, it is not responsible for target leakage prevention directly.

However, it provides timestamped raw data that enables point-in-time-aware feature engineering later in `features.py`.

For inference, `past_days` helps ensure that live features are built from recent context plus future forecast rows instead of selecting a later forecast row only because earlier rows lack lag or rolling history.

## Limitations / Future Work

* `event_time` is currently stored as timezone-naive local time returned by Open-Meteo for the configured timezone.
* A future version could add an explicit UTC-normalized event-time column.
* The project currently uses a single location, although the schema includes `location_id`.
* In this project, training uses historical actual weather data, while inference uses live forecast data. In production, it would be better to also store old forecast snapshots with their forecast creation time. This would reduce the difference between training data and live inference data.
