# src/backtesting_pipeline.py

## Role in the Project

`backtesting_pipeline.py` evaluates model performance across multiple chronological train/validation/test windows.

The goal is to check whether the model performs consistently over time instead of relying only on one fixed holdout split.

This pipeline is especially important for time-series forecasting because model performance can vary across seasons, weather regimes and time periods.

## Inputs

The pipeline currently uses the local feature cache created by `feature_pipeline.py`:

```text
data/features/temperature_features.parquet
```

Important config values:

* `FEATURE_CACHE_PATH`
* `BACKTESTING_RESULTS_PATH`
* `REPORTS_DIR`
* `TARGET_COLUMN`

It also uses the full feature schema from `features.py`:

```text
FEATURE_COLUMNS
```

Command-line arguments:

```text
--splits
--no-cache
```

`--splits` controls the number of walk-forward backtest windows.

`--no-cache` exists as a command-line option, but Hopsworks-based backtesting reads are currently unsupported. Backtesting currently expects the local feature cache.

## Outputs

The pipeline writes one local report artifact:

```text
reports/backtesting_results.csv
```

The report contains one row per backtest split with:

* split id
* train/validation/test date ranges
* row counts
* persistence baseline metrics
* seasonal naive 24h baseline metrics
* best naive baseline MAE
* model metrics
* MAE improvement over the best naive baseline

This artifact is a local report and is not committed to Git.

## Key Constants

### `DEFAULT_N_SPLITS`

Default number of walk-forward splits.

Current value:

```text
4
```

### `DEFAULT_INITIAL_TRAIN_FRACTION`

Fraction of the dataset used as the initial training window.

Current value:

```text
0.50
```

### `DEFAULT_VALID_FRACTION`

Fraction of the dataset used as the validation window in each split.

Current value:

```text
0.10
```

### `DEFAULT_TEST_FRACTION`

Fraction of the dataset used as the test window in each split.

Current value:

```text
0.10
```

## Key Functions

### `add_fixed_model_params()`

Returns fixed `HistGradientBoostingRegressor` parameters used during backtesting.

The current backtesting model uses the baseline-style model setup:

```text
loss = "squared_error"
max_iter = 200
learning_rate = 0.05
max_leaf_nodes = 31
min_samples_leaf = 20
l2_regularization = 0.0
early_stopping = False
random_state = 42
```

### `validate_training_dataframe()`

Validates and chronologically sorts the cached training dataframe.

It checks that the dataframe contains:

* the target column
* all expected feature columns
* `event_time`

### `load_training_dataframe()`

Loads the local Parquet feature cache.

Backtesting is currently implemented as an offline workflow and reads from:

```text
data/features/temperature_features.parquet
```

If `use_cache=False`, the function raises `NotImplementedError`.

### `split_features_and_target()`

Splits a dataframe into:

```text
X = feature columns
y = target column
```

### `create_backtest_windows()`

Creates expanding-window walk-forward backtest windows.

Each split consists of:

* training window
* immediately following validation window
* immediately following test window

The training window expands over time.

Conceptually:

```text
Split 1:
train: [early history]
valid: [next period]
test:  [following period]

Split 2:
train: [early history + more data]
valid: [next period]
test:  [following period]

Split N:
train: [all history available before that split]
valid: [next period]
test:  [following period]
```

The function validates that the requested number of splits and fractions fit into the available dataset.

### `evaluate_naive_baselines()`

Evaluates naive forecasting baselines on one test window:

* persistence baseline
* seasonal naive 24h baseline

The persistence baseline predicts that the future temperature equals the current temperature.

The seasonal naive 24h baseline predicts that the future temperature equals the temperature at the same feature timestamp one day before.

### `train_model()`

Trains one `HistGradientBoostingRegressor` for a backtest window.

### `evaluate_model()`

Evaluates the trained model on one backtest test split.

Metrics:

```text
model_mae
model_rmse
model_r2
```

### `evaluate_backtest_window()`

Runs the full evaluation for one backtest split.

It:

1. combines train and validation data
2. trains a model on train + validation
3. evaluates naive baselines on the test window
4. evaluates the model on the test window
5. computes improvement over the best naive baseline
6. returns metadata and metrics for the split

### `run_backtesting()`

Runs all walk-forward windows and returns one result dataframe.

It prints each split’s time ranges and key metrics.

### `save_backtesting_results()`

Saves the final backtesting results to:

```text
reports/backtesting_results.csv
```

### `main()`

Runs the complete backtesting workflow:

1. load local feature cache
2. create walk-forward windows
3. train and evaluate one model per window
4. print mean performance summary
5. save backtesting report

## Walk-Forward Backtesting

The pipeline uses expanding-window walk-forward backtesting.

This is stronger than evaluating only one fixed train/test split because it tests the model across several later time periods.

The training set grows over time, simulating repeated retraining as more historical data becomes available.

This helps answer:

```text
Does the model consistently beat simple baselines across time?
```

rather than only:

```text
Did the model perform well on one test period?
```

## Evaluation and Metrics

Each split reports:

```text
persistence_mae
persistence_rmse
persistence_r2
seasonal_naive_24h_mae
seasonal_naive_24h_rmse
seasonal_naive_24h_r2
best_naive_baseline_mae
model_mae
model_rmse
model_r2
mae_improvement_over_best_naive_baseline
```

The final summary reports mean values across all splits:

* mean model MAE
* mean best naive baseline MAE
* mean MAE improvement over the best naive baseline

MAE is the main metric because it directly represents average temperature error in degrees Celsius.

## Key Design Decisions

### Expanding Training Window

The training window grows over time.

This simulates a realistic production setup where a model can be retrained with all historical data available up to that point.

### Chronological Windows

Validation and test windows always occur after the training window.

This avoids future leakage.

### Local Offline Workflow

Backtesting currently uses the local feature cache only.

This makes it fast and reproducible for local experiments and avoids repeated Hopsworks reads.

### Baseline Comparison in Every Window

The model is compared against naive baselines in every test window.

This makes it clear whether the model improvement is stable over time.

### Train + Validation Fit per Window

For each backtest split, the model is trained on train + validation before evaluating on the test window.

This mirrors the final training approach.

## README-Relevant Points

* `backtesting_pipeline.py` performs expanding walk-forward backtesting.
* It evaluates model performance over multiple chronological windows.
* It uses the local feature cache.
* Each split trains a new model and evaluates on the next test period.
* The model is compared against persistence and seasonal naive baselines in every split.
* Results are saved to `reports/backtesting_results.csv`.
* Backtesting gives stronger evidence than a single holdout split.

## Limitations / Future Work

* Backtesting currently uses the full `FEATURE_COLUMNS`, not the final selected feature subset.
* Backtesting currently uses fixed baseline-style model parameters, not `best_params.json`.
* The pipeline currently reads only from the local feature cache.
* The default number of splits must fit into the available data size.
* A future version could backtest the exact final model setup with tuned parameters and selected features.
* A future version could support Hopsworks reads or versioned training datasets for backtesting.
