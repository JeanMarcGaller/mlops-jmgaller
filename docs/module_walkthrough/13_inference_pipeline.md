# src/inference_pipeline.py

## Role in the Project

`inference_pipeline.py` runs the live temperature forecasting workflow.

It loads the final model from the Hopsworks Model Registry, builds live inference features from Open-Meteo forecast data, writes the inference row to the Hopsworks Inference Feature Group, retrieves the same row through the Hopsworks Inference Feature View and produces a temperature prediction.

This pipeline is the serving/inference part of the Feature-Training-Inference architecture.

## Inputs

The pipeline uses:

- latest or configured model version from the Hopsworks Model Registry
- live Open-Meteo forecast data
- recent past forecast context from Open-Meteo
- feature engineering logic from `features.py`
- Hopsworks Inference Feature Group and Feature View
- selected feature order stored inside the model bundle

Important config values:

- `MODEL_NAME`
- `MODEL_VERSION`
- `TARGET_COLUMN`
- `FORECAST_HORIZON_HOURS`
- `TIMEZONE`
- `INFERENCE_FEATURE_GROUP_NAME`
- `INFERENCE_FEATURE_GROUP_VERSION`
- `INFERENCE_FEATURE_VIEW_NAME`
- `INFERENCE_FEATURE_VIEW_VERSION`
- `LATEST_PREDICTION_PATH`
- `PREDICTION_LOG_PATH`
- `REPORTS_DIR`

Local inference settings:

```text
INFERENCE_PAST_DAYS = 2
INFERENCE_FORECAST_DAYS = 2
```

## Outputs

The pipeline writes:

```text
reports/latest_prediction.json
reports/prediction_log.csv
```

It also writes the selected live inference feature row to the Hopsworks Inference Feature Group:

```text
temperature_forecast_inference_features
version 3
```

The prediction record contains:

* location id
* feature event time
* predicted event time
* target column
* predicted temperature
* resolved model version
* model test MAE
* model test RMSE
* MAE improvement over the best naive baseline

## Key Functions

### `get_latest_model_version()`

Returns the latest available model version for the configured model name from the Hopsworks Model Registry.

If no model exists for the given name, it raises a clear error.

### `load_model_from_registry()`

Loads the model bundle from the Hopsworks Model Registry.

If `MODEL_VERSION` is set in `.env`, that exact version is loaded.

If `MODEL_VERSION` is empty, the latest available model version is resolved automatically.

The function returns:

```text
model_bundle
resolved_model_version
```

The resolved version is later stored in the prediction record.

### `fetch_live_weather_context()`

Fetches recent past weather context and future forecast rows from Open-Meteo.

The recent past context is required for lag, rolling and trend features.

Without `past_days`, the first forecast rows would not have enough historical context to calculate features such as:

```text
temperature_lag_24h
temperature_mean_last_24h
pressure_trend_last_6h
```

The function also removes duplicate `location_id` and `event_time` rows and sorts the data chronologically.

### `build_current_inference_row()`

Builds the current live inference row.

It:

1. fetches live weather context
2. builds inference features with `include_target=False`
3. calculates the next full local forecast hour
4. selects the next usable forecast row
5. validates that the selected row is not later than the expected next forecast hour
6. returns one inference row

The target column is not created during inference because the future actual temperature is not known yet.

### `write_inference_features_to_hopsworks()`

Writes the selected inference feature row to the Hopsworks Inference Feature Group.

The Inference Feature Group is separate from the Training Feature Group because live inference rows do not yet have known target values.

The Feature Group uses:

```text
primary key: location_id, event_time_unix
event-time:  event_time
online:      enabled
```

### `get_or_create_inference_feature_view()`

Creates or loads the Hopsworks Inference Feature View.

This Feature View provides the serving-time access path used to retrieve the feature vector for prediction.

### `has_unnamed_columns()`

Checks whether a dataframe returned by Hopsworks appears to have unnamed positional columns.

This supports robust normalization of different Hopsworks feature vector response formats.

### `normalize_feature_vector()`

Converts a raw Hopsworks feature vector response into a dataframe.

Depending on Hopsworks client behavior, `get_feature_vector()` may return:

* a dataframe
* a dictionary
* a sequence-like object

The function safely normalizes the response and validates the schema.

It only renames columns when the response has unnamed positional columns and the number of columns matches the expected schema.

It refuses to rename already named but mismatched columns, because that could silently misalign feature values.

### `load_feature_vector_from_feature_view()`

Loads one inference feature vector from the Hopsworks Inference Feature View using:

```text
location_id
event_time_unix
```

It then normalizes and validates the returned feature vector.

### `prepare_model_input()`

Prepares the model input dataframe using the exact feature order stored in the model bundle.

This is critical because the model expects the same feature order that was used during training.

If required model input columns are missing, the function raises a clear error.

### `predict_temperature()`

Runs the model prediction and returns the predicted temperature as a float.

### `validate_model_bundle()`

Checks that the loaded model bundle contains the minimum required inference keys:

```text
model
feature_columns
```

### `build_prediction_record()`

Builds a JSON-serializable prediction record.

It includes both prediction output and model metadata such as resolved model version and test metrics.

### `save_latest_prediction()`

Writes the latest prediction to:

```text
reports/latest_prediction.json
```

### `append_prediction_log()`

Appends the prediction record to:

```text
reports/prediction_log.csv
```

This log is later used by the monitoring pipeline.

### `save_prediction_artifacts()`

Saves both the latest prediction JSON and the cumulative prediction log.

### `print_inference_result()`

Prints a human-readable inference summary, including:

* feature event time
* predicted event time
* predicted temperature
* model test metrics
* improvement over naive baseline

### `main()`

Runs the full live inference workflow:

1. load model from Hopsworks Model Registry
2. validate model bundle
3. build live inference row from Open-Meteo forecast data
4. write inference row to Hopsworks
5. retrieve feature vector through the Inference Feature View
6. prepare model input with the stored feature order
7. predict temperature
8. build prediction record
9. save prediction artifacts
10. print inference result

## Live Inference Flow

The live inference flow is:

```text
Open-Meteo forecast + past context
        ↓
feature engineering without target
        ↓
select next full forecast hour
        ↓
write row to Hopsworks Inference Feature Group
        ↓
retrieve row through Inference Feature View
        ↓
prepare model input with training feature order
        ↓
predict temperature
        ↓
save latest prediction and append prediction log
```

This demonstrates the inference part of the Feature-Training-Inference pattern.

## Hopsworks Integration

The pipeline integrates with Hopsworks in three ways:

### Model Registry

The final model bundle is loaded from:

```text
temperature_forecast_final_regressor
```

The pipeline can either load a configured model version or resolve the latest version automatically.

### Inference Feature Group

The selected live inference row is written to:

```text
temperature_forecast_inference_features
version 3
```

### Inference Feature View

The same row is retrieved through:

```text
temperature_forecast_inference_fv
version 3
```

This ensures that prediction uses the Feature View serving path instead of only the local dataframe.

## Model Version Handling

If `MODEL_VERSION` is set in `.env`, inference loads that exact version.

If `MODEL_VERSION` is empty, inference loads the latest available model version from Hopsworks.

The resolved model version is stored in every prediction record:

```text
model_version
```

This makes later monitoring and debugging easier because each prediction can be traced back to the model version that produced it.

## Feature Vector Validation

The pipeline validates the feature vector returned by Hopsworks before prediction.

This is important because feature vector responses can vary in structure depending on the Hopsworks client behavior.

The validation prevents silent schema misalignment.

In particular, the pipeline refuses to rename named columns when they do not match the expected schema.

This protects against incorrect predictions caused by features being passed to the model in the wrong order.

## Prediction Logging

The pipeline writes two local prediction artifacts:

```text
reports/latest_prediction.json
reports/prediction_log.csv
```

`latest_prediction.json` is useful for dashboards and quick inspection.

`prediction_log.csv` stores all local predictions over time and is used by the monitoring pipeline to compare predictions with later historical actual temperatures.

## Key Design Decisions

### Model Feature Order Comes from the Model Bundle

The inference pipeline does not hardcode the final feature list.

It loads `feature_columns` from the model bundle.

This ensures that inference uses exactly the same selected feature schema and order that were used during final training.

### Separate Inference Feature Group

Live inference rows are stored separately from historical training rows.

This avoids mixing rows with known targets and live rows where the future target is unknown.

### Recent Past Forecast Context

Inference fetches both recent past rows and future forecast rows.

This allows feature engineering to compute lag, rolling and trend features for the next forecast hour.

### Next Full Forecast Hour

The pipeline selects the next full forecast hour in the configured local timezone.

This avoids predicting for a timestamp that is already partially elapsed.

### Hopsworks Feature View Retrieval

The pipeline writes the inference row to Hopsworks and then retrieves it through the Feature View.

This makes the serving path explicit and closer to a real Feature Store inference workflow.

### Local Prediction Artifacts

Predictions are saved locally as JSON and CSV.

This keeps the project simple while still enabling later monitoring and dashboard visualization.

## README-Relevant Points

* `inference_pipeline.py` runs live temperature prediction.
* It loads the final model from the Hopsworks Model Registry.
* It supports either a pinned `MODEL_VERSION` or automatic latest-version loading.
* It fetches recent past and future forecast rows from Open-Meteo.
* It builds inference features with `include_target=False`.
* It selects the next full local forecast hour.
* It writes the inference row to the Hopsworks Inference Feature Group.
* It retrieves the feature vector through the Hopsworks Inference Feature View.
* It validates the feature vector schema before prediction.
* It uses the feature order stored in the model bundle.
* It saves `latest_prediction.json` and appends to `prediction_log.csv`.

## Limitations / Future Work

* The pipeline runs as a manual batch script, not as an always-on service.
* Hopsworks materialization can take time because `feature_group.insert(..., wait=True)` waits for completion.
* The local prediction log is file-based and not a production-grade monitoring store.
* The pipeline uses Open-Meteo forecast data at inference time, while training uses historical actual weather data.
* A production-grade setup could store historical forecast snapshots with forecast issue times to reduce training-serving skew.
* A future version could run inference on a schedule, for example through GitHub Actions, cron or an orchestrator.
* A future version could persist predictions and monitoring results in Hopsworks or a database instead of local CSV files.
