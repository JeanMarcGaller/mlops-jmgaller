# src/baseline_training_pipeline.py

## Role in the Project

`baseline_training_pipeline.py` trains the first supervised machine learning model for the temperature forecasting project.

It establishes a reproducible baseline before more advanced steps such as hyperparameter tuning, feature selection and final model training.

The pipeline uses historical engineered features from the Training Feature Group or from the local feature cache and trains a `HistGradientBoostingRegressor`.

## Inputs

The pipeline uses:

- engineered historical training features
- the shared feature schema from `features.py`
- the target column from `config.py`
- Hopsworks Training Feature Group and Feature View configuration
- optional local Parquet feature cache
- optional command-line arguments

Important config values:

- `FEATURE_CACHE_PATH`
- `MODEL_DIR`
- `MODEL_LOCAL_PATH`
- `TARGET_COLUMN`
- `BASELINE_MODEL_NAME`
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
temperature_forecast_baseline_regressor
```

The saved model bundle contains:

* trained model object
* model name
* feature column order
* target column
* metrics
* model type
* model parameters
* Feature Group metadata
* Feature View metadata

## Key Functions

### `get_or_create_training_feature_view()`

Creates or loads the Hopsworks Training Feature View.

The Feature View defines a reusable training interface over the Training Feature Group.

In this project, the actual dataframe read is done directly from the Feature Group with robust read logic. The Feature View is still created to keep the Feature-Training-Inference architecture explicit in Hopsworks.

### `validate_training_dataframe()`

Validates and sorts the loaded training dataframe.

It checks that the dataframe contains:

* the target column
* all expected feature columns
* `event_time`

It converts `event_time` to datetime and sorts rows chronologically.

### `load_training_dataframe()`

Loads training data either from:

* local Parquet cache, if `--use-cache` is set
* Hopsworks Feature Group, otherwise

When reading from Hopsworks, it uses `read_feature_group_with_retry()` from `hopsworks_read_utils.py`, which adds retry behavior and local cache fallback.

### `time_based_train_valid_test_split()`

Splits the dataframe chronologically into:

* training set
* validation set
* test set

Default fractions:

```text
train: 70%
valid: 15%
test:  15%
```

The split is time-based, not random. This avoids leaking future information into the training set.

### `split_features_and_target()`

Separates model inputs from the target.

It returns:

```text
X = FEATURE_COLUMNS
y = TARGET_COLUMN
```

### `evaluate_naive_baselines()`

Evaluates simple naive forecasting baselines on the test set.

The pipeline uses:

* persistence baseline
* seasonal naive 24h baseline

The persistence baseline predicts that the future temperature equals the current temperature.

The seasonal naive 24h baseline predicts that the future temperature equals the temperature at the same feature timestamp one day before.

### `train_model()`

Trains the initial baseline ML model.

The model is a `HistGradientBoostingRegressor` with fixed parameters.

This model is strong enough for tabular lag, rolling and calendar features while still being simple and reproducible.

### `evaluate_model()`

Evaluates the trained model on validation and test data.

Metrics:

* MAE
* RMSE
* R2

Validation metrics are used to inspect model behavior during development. Test metrics estimate performance on unseen later data.

### `combine_metrics()`

Combines model metrics and naive baseline metrics.

It also calculates:

```text
mae_improvement_over_best_naive_baseline
```

This shows how much the ML model improves over the stronger naive baseline.

### `build_model_bundle()`

Creates the serializable model bundle.

The bundle stores not only the model object, but also metadata required by inference, especially the exact feature column order.

This is important because inference must pass features to the model in the same order as training.

### `save_model_locally()`

Saves the model bundle locally as:

```text
models/model.joblib
```

### `upload_model_to_registry()`

Uploads the locally saved model bundle to the Hopsworks Model Registry.

The baseline model is uploaded under the dedicated registry name:

```text
temperature_forecast_baseline_regressor
```

It also writes a `metrics.json` file into the model export directory.

### `main()`

Runs the full baseline training workflow:

1. optionally connects to Hopsworks
2. creates or loads the Training Feature View
3. loads training data
4. creates a chronological split
5. evaluates naive baselines
6. trains the baseline model
7. evaluates the model
8. combines metrics
9. saves the model locally
10. optionally uploads the model to Hopsworks

## Key Design Decisions

### Chronological Split

The data is split by time instead of randomly.

This is essential for time-series forecasting because random splits can leak information from the future into the training set.

### Naive Baseline Comparison

The ML model is evaluated against simple baselines.

This prevents overvaluing a machine learning model that does not actually improve over simple forecasting rules.

### Separate Baseline Model Registry Name

The baseline model uses its own registry name:

```text
temperature_forecast_baseline_regressor
```

This prevents a newly uploaded baseline model from accidentally becoming the latest model used by the inference pipeline.

The inference pipeline uses the final model registry name by default, not the baseline registry name.

### Local Cache Mode

`--use-cache` allows training without reading from Hopsworks.

This is useful for local experiments, faster iteration and situations where Hopsworks offline reads are temporarily unstable.

### Optional Registry Upload

`--skip-upload` allows local training runs without uploading the model to Hopsworks.

This is useful during development and testing.

### Model Bundle Metadata

The model is saved together with metadata such as feature columns, target column, metrics and Feature Store versions.

This makes inference more reproducible and less dependent on hardcoded assumptions.

## Evaluation and Metrics

The pipeline reports metrics for:

* naive baselines
* validation data
* test data
* improvement over the best naive baseline

Metrics used:

```text
MAE
RMSE
R2
```
MAE is the main metric because it directly represents the average forecast error in degrees Celsius. This makes it easy to interpret and compare against naive baselines.

RMSE is reported as a complementary error metric because it penalizes larger errors more strongly than MAE. It helps detect whether occasional larger forecast errors occur.

R² is reported as an additional goodness-of-fit metric. It shows how much of the target variance is explained by the model compared with a simple mean prediction. However, MAE remains the primary metric because it is easier to interpret in the original unit of the problem: degrees Celsius.

## Hopsworks Integration

The pipeline integrates with Hopsworks in two ways:

1. Feature Store:

   * creates or loads the Training Feature View
   * reads training data from the Training Feature Group

2. Model Registry:

   * uploads the trained baseline model bundle
   * stores metrics together with the model artifact

The Training Feature Group and Feature View versions come from `config.py`.

## Limitations / Future Work

* The model parameters are fixed and not tuned in this pipeline.
* Feature selection is not applied here.
* The pipeline is intended as an initial ML baseline, not the final production candidate.
* The Feature View is created for architecture completeness, while robust dataframe reads are currently performed directly from the Feature Group.
* For a more production-like training setup, this pipeline could log experiment metadata in a dedicated experiment tracker.
