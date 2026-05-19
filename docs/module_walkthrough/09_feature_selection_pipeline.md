# src/feature_selection_pipeline.py

## Role in the Project

`feature_selection_pipeline.py` selects a smaller and more relevant set of model features after hyperparameter tuning.

It uses the best Optuna parameters from `tuning_pipeline.py`, trains a tuned `HistGradientBoostingRegressor`, computes diagnostic correlation reports, adds random probe features and then calculates permutation importance on the validation split.

The selected features are later used by the final training pipeline.

## Inputs

The pipeline uses:

- engineered historical training features
- the full feature schema from `features.py`
- best hyperparameters from `reports/best_params.json`
- Hopsworks Training Feature Group data or the local feature cache
- command-line argument `--use-cache`

Important config values:

- `BEST_PARAMS_PATH`
- `FEATURE_CACHE_PATH`
- `PERMUTATION_IMPORTANCE_PATH`
- `SELECTED_FEATURES_PATH`
- `FEATURE_CORRELATION_MATRIX_PATH`
- `HIGH_CORRELATION_PAIRS_PATH`
- `TARGET_CORRELATION_PATH`
- `REPORTS_DIR`
- `TARGET_COLUMN`
- `TRAINING_FEATURE_GROUP_NAME`
- `TRAINING_FEATURE_GROUP_VERSION`

## Outputs

The pipeline writes local feature-selection and diagnostic artifacts:

```text
reports/permutation_importance.csv
reports/selected_features.json
reports/feature_correlation_matrix.csv
reports/highly_correlated_features.csv
reports/target_correlation.csv
```

`selected_features.json` is the most important output because it is consumed by `final_training_pipeline.py`.

It contains:

* all original features
* selected features
* number of selected features
* selection method
* random probe threshold
* random probe importances
* best Optuna parameters
* final model parameters
* validation metrics
* correlation threshold metadata

These artifacts are experiment reports and are not committed to Git.

## Key Constants

### `MIN_SELECTED_FEATURES`

Defines the minimum number of real features to keep.

If fewer than this number outperform the random probe threshold, the pipeline keeps the top real features by permutation importance.

### `RANDOM_FEATURE_COLUMNS`

Defines temporary random probe features:

```text
random_normal_probe
random_uniform_probe
random_permuted_temperature_probe
```

These features intentionally contain no useful signal. They are used only inside this pipeline to estimate how important pure noise can appear.

### `CORRELATION_THRESHOLD`

Defines the threshold for reporting highly correlated feature pairs.

Current value:

```text
0.90
```

## Key Functions

### `add_fixed_model_params()`

Adds fixed `HistGradientBoostingRegressor` settings to the best Optuna parameters.

Fixed parameters:

```text
loss = "squared_error"
early_stopping = False
random_state = 42
```

This keeps feature selection consistent with tuning and final training.

### `validate_training_dataframe()`

Validates and chronologically sorts the loaded training dataframe.

It checks that the dataframe contains:

* the target column
* all expected feature columns
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

Feature selection uses the validation set. The test set is not used for feature selection.

### `split_features_and_target()`

Splits a dataframe into selected model input features and the target column.

Unlike earlier pipelines, this function accepts `feature_columns` as an argument because feature selection trains with both real features and temporary random probe features.

### `add_random_probe_features()`

Adds artificial random features to the dataframe.

The random probes are:

* normally distributed random noise
* uniformly distributed random noise
* a randomly permuted version of `temperature_2m`

These probes are used as a noise benchmark for permutation importance.

Random probe features are never written to Hopsworks, never used for inference and never included as final selected features.

### `compute_correlation_reports()`

Computes diagnostic correlation reports:

1. full feature correlation matrix
2. highly correlated feature pairs
3. target correlation ranking

These reports support interpretation and documentation, but they are not the direct feature-selection rule.

### `save_correlation_reports()`

Saves correlation diagnostics to local CSV files:

```text
feature_correlation_matrix.csv
highly_correlated_features.csv
target_correlation.csv
```

### `load_best_params()`

Loads the best Optuna hyperparameters from:

```text
reports/best_params.json
```

If the file is missing, the pipeline raises a clear error and asks the user to run the tuning pipeline first.

### `train_model()`

Trains a `HistGradientBoostingRegressor` using the best Optuna parameters plus fixed model settings.

### `evaluate_model()`

Evaluates the tuned model on the validation split before feature selection.

Metrics:

```text
MAE
RMSE
R2
```

### `compute_permutation_importance()`

Computes permutation importance on the validation split.

The scoring metric is:

```text
neg_mean_absolute_error
```

Permutation importance measures how much validation performance changes when one feature is randomly shuffled. If shuffling a feature worsens performance, the feature is likely useful.

### `get_random_probe_threshold()`

Finds the strongest permutation importance score among the random probe features.

This value becomes the noise threshold for feature selection.

### `select_relevant_features()`

Selects real features whose permutation importance is higher than the strongest random probe.

If too few real features pass this threshold, the pipeline keeps the top `MIN_SELECTED_FEATURES` real features as a fallback.

### `build_selected_features_payload()`

Builds the JSON payload for `selected_features.json`.

It stores selected features, random probe metadata, best parameters, model parameters and validation metrics.

### `save_feature_selection_artifacts()`

Saves:

```text
reports/permutation_importance.csv
reports/selected_features.json
```

### `main()`

Runs the full feature-selection workflow:

1. load training data
2. create chronological train/validation/test split
3. load best Optuna parameters
4. compute correlation diagnostics
5. add random probe features
6. train tuned model with real and probe features
7. evaluate validation performance
8. compute permutation importance
9. select relevant real features
10. save feature-selection artifacts

## Feature Selection Method

The feature-selection method is:

```text
permutation_importance_with_random_probes
```

The idea is to compare real feature importance against artificial noise features.

A real feature is selected directly if:

```text
real feature importance > strongest random probe importance
```

This avoids keeping features that perform no better than noise.

If too few features pass this rule, the pipeline keeps the top real features by permutation importance as a fallback.

## Diagnostics and Reports

### Correlation Reports

Correlation reports help interpret the feature space.

They show:

* which features are strongly correlated with each other
* which features have the strongest linear relationship with the target
* where redundant feature groups may exist

The target correlation report showed that temperature-history features have the strongest linear relationship with the 6-hour-ahead target.

The strongest feature was:

```text
temperature_lag_18h
```

This is meteorologically plausible. For a 6-hour forecast horizon, `temperature_lag_18h` represents approximately the temperature at the same target hour on the previous day.

Example:

```text
feature_event_time:   2026-05-13 07:00
predicted_event_time: 2026-05-13 13:00
temperature_lag_18h:  2026-05-12 13:00
```

Other highly correlated features include:

```text
temperature_mean_last_24h
temperature_2m
temperature_lag_1h
temperature_lag_24h
temperature_mean_last_6h
temperature_lag_3h
temperature_lag_6h
```

Seasonal cyclic features such as `month_cos` also show strong target correlation, reflecting annual temperature seasonality.

Correlation reports are diagnostic only. They are not used directly for feature selection because correlation can miss nonlinear effects and can overstate redundant features.

### Permutation Importance Report

Permutation importance is the actual model-based feature-selection signal.

It evaluates how much validation performance changes when a feature is shuffled.

This report includes both real features and random probe features.

## Key Design Decisions

### Validation Set Only

Feature selection is performed on the validation set.

The test set remains untouched and is reserved for final evaluation.

### Random Probe Calibration

Random probe features provide a noise benchmark.

This helps avoid selecting features that only appear useful due to random variation.

### Temporary Random Probes

Random probe features exist only inside this pipeline.

They are not written to Hopsworks, not used by inference and not included in the final selected feature list.

### Correlation Reports Are Diagnostic

Correlation reports are saved for interpretation and documentation.

The actual selection rule is based on permutation importance with random probes.

### Local Cache Mode

`--use-cache` allows feature selection without reading from Hopsworks.

This supports faster local experimentation and makes the workflow robust against temporary Hopsworks read issues.

## Evaluation and Metrics

The tuned model is evaluated on the validation split before feature selection.

Metrics:

```text
MAE
RMSE
R2
```

The main feature-selection signal is permutation importance with `neg_mean_absolute_error`.

MAE remains the most interpretable metric because it directly represents temperature error in degrees Celsius.

## README-Relevant Points

* `feature_selection_pipeline.py` selects features after Optuna tuning.
* It loads `best_params.json`.
* It computes correlation reports for diagnostics.
* Temperature-history features dominate the target correlation report.
* `temperature_lag_18h` is especially meaningful because it approximates the previous day at the target hour.
* It adds random probe features as a noise benchmark.
* It computes permutation importance on the validation split.
* It selects real features that outperform random probes.
* It saves `selected_features.json`.
* Random probe features are temporary and never used for inference.
* The test set is not used for feature selection.

## Limitations / Future Work

* Feature selection is based on one chronological validation split.
* Selected features may vary if the validation period changes.
* Permutation importance can be affected by correlated features.
* Correlation reports are diagnostic only and do not automatically remove redundant features.
* A future version could perform feature selection across multiple walk-forward splits.
* A future version could compare feature-selection stability across seasons.
