# src/monitoring_pipeline.py

## Role in the Project

`monitoring_pipeline.py` evaluates logged temperature predictions after the predicted timestamps become historically available.

The inference pipeline creates predictions and appends them to:

```text
reports/prediction_log.csv
```

The monitoring pipeline later reads this log, fetches actual historical temperatures from Open-Meteo and computes forecast errors.

This closes the loop from prediction to observed outcome.

## Inputs

The pipeline uses the local prediction log:

```text
reports/prediction_log.csv
```

It also calls the Open-Meteo Historical Weather API through `weather_api.py` to retrieve actual temperatures for predicted timestamps.

Important config values:

* `PREDICTION_LOG_PATH`
* `MONITORING_RESULTS_PATH`
* `REPORTS_DIR`

The prediction log must contain:

```text
location_id
feature_event_time
predicted_event_time
predicted_temperature_celsius
```

## Outputs

The pipeline writes one local monitoring artifact:

```text
reports/monitoring_results.csv
```

The monitoring output contains the original prediction records plus:

```text
actual_temperature_celsius
error_celsius
absolute_error_celsius
squared_error_celsius
rolling_mae_7_predictions
```

This artifact is used by the dashboard and for project reporting.

## Key Functions

### `load_prediction_log()`

Loads the local prediction log from:

```text
reports/prediction_log.csv
```

It validates that the file exists, is not empty and contains the required columns.

It converts:

```text
feature_event_time
predicted_event_time
```

to pandas datetime values.

### `fetch_actual_temperatures_for_predictions()`

Fetches actual historical temperatures for predicted timestamps.

The function first checks which predictions are old enough to evaluate.

It only tries to monitor predictions where:

```text
predicted_event_time date <= yesterday
```

This avoids requesting actual values for predictions that are still in the future or not yet available in the historical API.

It then fetches historical Open-Meteo data for the required date range and returns:

```text
location_id
predicted_event_time
actual_temperature_celsius
```

### `build_monitoring_results()`

Joins the prediction log with actual temperatures.

The join key is:

```text
location_id
predicted_event_time
```

This is important because predictions should be matched to the correct location and predicted target timestamp.

The function then computes:

```text
error_celsius = predicted_temperature_celsius - actual_temperature_celsius
absolute_error_celsius = abs(error_celsius)
squared_error_celsius = error_celsius ** 2
```

It also computes a rolling 7-prediction MAE:

```text
rolling_mae_7_predictions
```

### `summarize_monitoring_results()`

Computes monitoring summary metrics:

```text
n_matched_predictions
monitoring_mae
monitoring_rmse
latest_absolute_error_celsius
```

These metrics are printed in the terminal after each monitoring run.

### `save_monitoring_results()`

Saves the monitoring dataframe to:

```text
reports/monitoring_results.csv
```

### `main()`

Runs the complete monitoring workflow:

1. load prediction log
2. fetch actual historical temperatures
3. join predictions with actuals
4. compute forecast errors
5. print monitoring summary
6. save monitoring results

If no predictions are old enough yet, or if no rows can be matched, monitoring is skipped cleanly with a readable message.

## Monitoring Flow

The monitoring flow is:

```text
prediction_log.csv
        ↓
filter predictions old enough for historical actuals
        ↓
fetch actual temperatures from Open-Meteo Historical API
        ↓
join by location_id and predicted_event_time
        ↓
compute forecast errors
        ↓
save monitoring_results.csv
```

This makes monitoring report-based and simple, while still connecting predictions to later observed outcomes.

## Evaluation and Metrics

The pipeline calculates per-prediction errors:

```text
error_celsius
absolute_error_celsius
squared_error_celsius
```

It also calculates aggregate monitoring metrics:

```text
Monitoring MAE
Monitoring RMSE
Latest absolute error
Rolling MAE over last 7 matched predictions
```

MAE is the most interpretable metric because it directly represents average temperature error in degrees Celsius.

RMSE penalizes larger errors more strongly.

The rolling MAE gives a small local trend of recent prediction quality.

## Key Design Decisions

### Local Report-Based Monitoring

Monitoring uses local CSV artifacts instead of a database.

This keeps the project simple and transparent while still demonstrating a complete prediction-to-monitoring loop.

### Historical Actuals Only

A prediction can only be monitored after its predicted timestamp is available through the historical weather API.

If the prediction is too recent, the pipeline skips monitoring instead of failing.

### Join by Location and Predicted Time

Predictions are matched with actual temperatures using both:

```text
location_id
predicted_event_time
```

This avoids incorrect matches when multiple locations or repeated timestamps exist.

### Non-Destructive Monitoring

The pipeline does not modify the prediction log.

It reads the log and writes a separate monitoring result file.

### Rolling Error Metric

The rolling 7-prediction MAE gives a simple view of recent model performance.

This is useful once multiple live predictions have been collected.

## README-Relevant Points

* `monitoring_pipeline.py` evaluates logged predictions after actual temperatures become available.
* It reads `reports/prediction_log.csv`.
* It fetches historical actual temperatures from Open-Meteo.
* It joins predictions with actuals by `location_id` and `predicted_event_time`.
* It computes forecast errors and monitoring metrics.
* It writes `reports/monitoring_results.csv`.
* Monitoring is skipped cleanly if predictions are still too recent.
* The monitoring setup is local and report-based.

## Limitations / Future Work

* Monitoring depends on local `prediction_log.csv`.
* Monitoring results are stored in local CSV only.
* The pipeline currently fetches historical actuals from Open-Meteo, not from a production observation database.
* The availability rule uses yesterday as the latest historical date, which is simple but conservative.
* A production version could persist predictions and actuals in Hopsworks or a database.
* A production version could add alerting when MAE or latest error exceeds a threshold.
* A production version could monitor data drift and feature drift in addition to prediction error.
