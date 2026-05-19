# src/final_training_pipeline.py

## Role in the Project

`final_training_pipeline.py` trains the final production candidate model for the temperature forecasting project.

It combines the outputs of the previous modeling stages:

- best hyperparameters from `tuning_pipeline.py`
- selected features from `feature_selection_pipeline.py`
- engineered historical training data from the Training Feature Group or local cache

The final model is trained on the training and validation periods and evaluated once on the held-out test period.

## Inputs

The pipeline uses:

- engineered historical training features
- best Optuna parameters from `reports/best_params.json`
- selected features from `reports/selected_features.json`
- Hopsworks Training Feature Group data or the local feature cache
- command-line arguments

Important config values:

- `BEST_PARAMS_PATH`
- `SELECTED_FEATURES_PATH`
- `FEATURE_CACHE_PATH`
- `MODEL_DIR`
- `MODEL_LOCAL_PATH`
- `MODEL_NAME`
- `TARGET_COLUMN`
- `TRAINING_FEATURE_GROUP_NAME`
- `TRAINING_FEATURE_GROUP_VERSION`
- `TRAINING_FEATURE_VIEW_NAME`
- `TRAINING_FEATURE_VIEW_VERSION`

Command-line arguments:

```text
--use-cache
--skip-upload
```

`--use-cache` loads training data from the local Parquet cache instead of Hopsworks.

`--skip-upload` trains and saves the model locally but skips the Hopsworks Model Registry upload.

## Outputs

The pipeline produces:

1. A local model bundle:

```text
models/model.joblib
```

2. Optionally, a Hopsworks Model Registry artifact:

```text
temperature_forecast_final_regressor
```

The model export directory also contains:

```text
model.joblib
metrics.json
selected_features.json
```

The local `model.joblib` bundle contains:

* trained model object
* final model name
* model stage
* selected feature order
* target column
* metrics
* model type
* model parameters
* best Optuna parameters
* feature selection metadata
* Feature Group metadata
* Feature View metadata

## Key Functions

### `add_fixed_model_params()`

Adds fixed `HistGradientBoostingRegressor` parameters to the tuned Optuna parameters.

Fixed parameters:

```text
loss = "squared_error"
early_stopping = False
random_state = 42
```

These settings keep final training reproducible and consistent with tuning and feature selection.

### `load_best_params()`

Loads the best hyperparameters from:

```text
reports/best_params.json
```

If the file is missing, the pipeline raises a clear error and asks the user to run the tuning pipeline first.

### `load_selected_features()`

Loads the selected feature list from:

```text
reports/selected_features.json
```

If the file is missing, the pipeline raises a clear error and asks the user to run the feature-selection pipeline first.

The selected feature order is important because the inference pipeline must provide model inputs in exactly the same order.

### `validate_training_dataframe()`

Validates and chronologically sorts the training dataframe.

It checks that the dataframe contains:

* the target column
* all selected feature columns
* `event_time`

### `load_training_dataframe()`

Loads training data either from:

* the local Parquet cache if `--use-cache` is set
* the Hopsworks Training Feature Group otherwise

When reading from Hopsworks, it uses robust read logic from `hopsworks_read_utils.py`.

### `time_based_train_valid_test_split()`

Splits data chronologically into:

* training set
* validation set
* test set

The final model is trained on train + validation and evaluated on the held-out test set.

### `split_features_and_target()`

Splits a dataframe into:

```text
X = selected feature columns
y = target column
```

### `train_final_model()`

Trains the final `HistGradientBoostingRegressor`.

It concatenates the training and validation splits and fits the model on:

```text
train + validation
```

This uses more data for the final model while keeping the test split untouched for final evaluation.

### `evaluate_naive_baselines()`

Evaluates naive forecasting baselines on the test set:

* persistence baseline
* seasonal naive 24h baseline

The persistence baseline predicts that the future temperature equals the current temperature.

The seasonal naive baseline predicts that the future temperature equals the temperature at the same feature timestamp one day before.

### `evaluate_final_model()`

Evaluates the final trained model on the held-out test set.

Metrics:

```text
MAE
RMSE
R2
```

### `combine_metrics()`

Combines final model metrics, naive baseline metrics and selected-feature metadata.

It also computes:

```text
mae_improvement_over_best_naive_baseline
```

This shows how much the final model improves over the stronger naive baseline.

### `build_model_bundle()`

Builds the serializable model bundle stored as `model.joblib`.

This bundle contains the model object and all metadata needed by the inference pipeline, especially the exact selected feature order.

### `save_model_locally()`

Saves the final model bundle locally to:

```text
models/model.joblib
```

### `upload_model_to_registry()`

Uploads the final model artifact to the Hopsworks Model Registry.

The final model is uploaded under:

```text
temperature_forecast_final_regressor
```

The export directory includes the model bundle, metrics and selected feature list.

### `main()`

Runs the complete final training workflow:

1. load best Optuna parameters
2. load selected features
3. load historical training data
4. create chronological train/validation/test split
5. train final model on train + validation
6. evaluate naive baselines on test
7. evaluate final model on test
8. compare final model against the best naive baseline
9. save model locally
10. optionally upload model to Hopsworks Model Registry

## Final Model Training

The final model uses:

* tuned hyperparameters from Optuna
* selected features from feature selection
* fixed reproducibility parameters
* train + validation data for fitting
* held-out test data for final evaluation

This makes the final training stage the production candidate step of the project.

## Evaluation and Metrics

The pipeline reports:

* persistence baseline MAE, RMSE and R2
* seasonal naive 24h baseline MAE, RMSE and R2
* final model test MAE, RMSE and R2
* improvement over the best naive baseline
* number of selected features

MAE is the main metric because it directly represents the average temperature error in degrees Celsius.

## Hopsworks Integration

The pipeline integrates with Hopsworks in two ways:

1. Feature Store:

   * reads historical training data from the Training Feature Group

2. Model Registry:

   * uploads the final model artifact
   * stores metrics with the model
   * stores selected-feature metadata in the model export directory

The final model uses the registry name defined by `MODEL_NAME`, which points to:

```text
temperature_forecast_final_regressor
```

## Key Design Decisions

### Final Model Uses Selected Features

The final model does not use all engineered features.

It uses the selected feature list from `selected_features.json`, which was produced by permutation importance with random probe calibration.

### Feature Order Stored in Model Bundle

The exact selected feature order is stored in the model bundle.

This is critical because the inference pipeline must construct model input in the same order used during training.

### Train + Validation for Final Fit

After tuning and feature selection, the final model is trained on both the training and validation periods.

The test period remains held out for final evaluation.

### Separate Final Model Registry Name

The final model uses a separate registry name from the baseline model.

This prevents baseline artifacts from being loaded accidentally as production candidates.

### Local-Only Mode

`--skip-upload` allows local final training without uploading to Hopsworks.

This is useful for debugging and local experiments.

### Cache Mode

`--use-cache` allows final training without reading from Hopsworks.

This supports faster local iteration and robustness against temporary Hopsworks read issues.

## README-Relevant Points

* `final_training_pipeline.py` trains the production candidate model.
* It loads `best_params.json` from tuning.
* It loads `selected_features.json` from feature selection.
* It trains on train + validation data.
* It evaluates once on the held-out test set.
* It compares against persistence and seasonal naive baselines.
* It saves a local model bundle.
* It can upload the final model to Hopsworks Model Registry.
* The final model registry name is `temperature_forecast_final_regressor`.
* The selected feature order is stored in the model bundle for inference reproducibility.

## Limitations / Future Work

* Final training uses one fixed chronological split.
* Broader temporal robustness is assessed separately by the backtesting pipeline.
* The pipeline does not retrain automatically on a schedule.
* Model metadata is stored in the model bundle and registry artifact, but not in a dedicated experiment tracking system.
* A future version could register model stages such as staging, production or archived.
* A future version could compare several tuned model families before choosing the final production candidate.
