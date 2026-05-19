# src/tuning_pipeline.py

## Role in the Project

`tuning_pipeline.py` performs hyperparameter tuning for the main regression model.

It tunes a `HistGradientBoostingRegressor` with Optuna and saves the best parameter set for later use by the feature selection and final training pipelines.

The pipeline comes after the baseline training pipeline. Its purpose is to improve model performance while keeping the test set untouched during the search process.

## Inputs

The pipeline uses:

- engineered historical training features
- the shared feature schema from `features.py`
- the target column from `config.py`
- Hopsworks Training Feature Group data or the local feature cache
- command-line tuning settings

Important config values:

- `FEATURE_CACHE_PATH`
- `BEST_PARAMS_PATH`
- `TUNING_RESULTS_PATH`
- `REPORTS_DIR`
- `TARGET_COLUMN`
- `TRAINING_FEATURE_GROUP_NAME`
- `TRAINING_FEATURE_GROUP_VERSION`

Command-line arguments:

```text
--trials
--use-cache
```

`--trials` controls the number of Optuna trials.

`--use-cache` loads training data from the local Parquet cache instead of Hopsworks.

## Outputs

The pipeline writes two local report artifacts:

```text
reports/best_params.json
reports/tuning_results.csv
```

`best_params.json` contains:

* model type
* best hyperparameters
* final model parameters including fixed settings
* best validation MAE
* test evaluation metrics for the best tuned model
* feature columns
* target column

`tuning_results.csv` contains the full Optuna trial history.

These files are experiment artifacts and are not committed to Git.

## Key Functions

### `add_fixed_model_params()`

Adds fixed `HistGradientBoostingRegressor` parameters to a dictionary of tunable parameters.

Fixed parameters:

```text
loss = "squared_error"
early_stopping = False
random_state = 42
```

These settings are kept constant across all trials for reproducibility.

### `load_training_dataframe()`

Loads training data either from:

* the local Parquet cache if `--use-cache` is set
* the Hopsworks Training Feature Group otherwise

When reading from Hopsworks, it uses robust read logic from `hopsworks_read_utils.py`, including retries and local cache fallback.

### `validate_training_dataframe()`

Validates and sorts the loaded dataframe.

It checks that the dataframe contains:

* the target column
* all expected feature columns
* `event_time`

It then converts `event_time` to datetime and sorts rows chronologically.

### `time_based_train_valid_test_split()`

Creates a chronological train/validation/test split.

Default fractions:

```text
train: 70%
valid: 15%
test:  15%
```

The validation split is used by Optuna to compare hyperparameter trials.

The test split is held back and used only once after tuning.

### `split_features_and_target()`

Separates the dataframe into:

```text
X = FEATURE_COLUMNS
y = TARGET_COLUMN
```

### `create_objective()`

Creates the Optuna objective function.

For each trial, Optuna samples hyperparameters, trains a `HistGradientBoostingRegressor` on the training split and returns the validation MAE.

Optuna minimizes this validation MAE.

### `run_optuna_tuning()`

Runs the Optuna study.

The pipeline uses a `TPESampler` with a fixed seed for reproducibility.

The study direction is:

```text
minimize
```

because lower validation MAE is better.

### `train_best_model_and_evaluate()`

After tuning, the best hyperparameters are used to train a model on:

```text
train + validation
```

The resulting model is then evaluated once on the held-out test set.

This gives a final estimate of tuned model performance without using the test set during the search.

### `build_best_params_payload()`

Builds the JSON-serializable payload for `best_params.json`.

It stores both the raw best Optuna parameters and the final model parameters including fixed training settings.

### `save_tuning_artifacts()`

Saves tuning artifacts locally:

```text
reports/best_params.json
reports/tuning_results.csv
```

### `main()`

Runs the full tuning workflow:

1. validate number of trials
2. load training data
3. create chronological splits
4. split train and validation into features and target
5. run Optuna tuning
6. train best model on train + validation
7. evaluate once on test data
8. save tuning artifacts

## Hyperparameter Search

Optuna tunes these `HistGradientBoostingRegressor` parameters:

```text
max_iter
learning_rate
max_leaf_nodes
min_samples_leaf
l2_regularization
max_bins
```

The following parameters are fixed:

```text
loss = "squared_error"
early_stopping = False
random_state = 42
```

This keeps the search space focused while making experiments reproducible.

## Evaluation and Metrics

The tuning objective is:

```text
validation MAE
```

The best model is finally evaluated on the test set with:

```text
MAE
RMSE
R2
```

MAE is the main metric because it directly represents the average temperature error in degrees Celsius. (Cf. 07_baseline_training_pipeline.md)

## Key Design Decisions

### Test Set Is Not Used During Search

The test set is not used by Optuna.

It is only used once after tuning to estimate generalization performance of the selected parameter set.

### Chronological Split

The pipeline uses a time-based split instead of a random split.

This is important for time-series forecasting because future observations should not influence training or hyperparameter selection.

### Reproducible Tuning

The Optuna sampler uses a fixed seed.

The model also uses a fixed `random_state`.

This makes tuning runs more reproducible.

### Local Cache Mode

`--use-cache` enables local tuning without reading from Hopsworks.

This is useful for faster experimentation and for cases where Hopsworks offline reads are temporarily unstable.

### Artifacts Feed Later Pipelines

`best_params.json` is consumed by later pipelines:

* feature selection
* final training

This makes tuning a separate, reusable pipeline stage.



## Limitations / Future Work

* The pipeline tunes only `HistGradientBoostingRegressor`.
* Other models such as XGBoost, LightGBM or CatBoost are not used and tuned here.
* The search space is manually defined and relatively compact.
* The study is stored only as local CSV/JSON artifacts, not in a dedicated experiment tracking system.
* The test set is evaluated once after tuning, but broader performance stability is assessed separately by the backtesting pipeline.
