# src/model_comparison_pipeline.py

## Role in the Project

`model_comparison_pipeline.py` compares multiple regression model families on the same chronological train/validation/test split.

It is a report-oriented pipeline. It does not upload models to the Hopsworks Model Registry and does not change the final production model directly. Its purpose is to provide evidence for whether the chosen model family is reasonable compared with alternatives.

## Inputs

The pipeline uses:

- engineered historical training features
- the full feature schema from `features.py`
- Hopsworks Training Feature Group data or the local feature cache
- command-line argument `--use-cache`

Important config values:

- `FEATURE_CACHE_PATH`
- `MODEL_COMPARISON_RESULTS_PATH`
- `REPORTS_DIR`
- `TARGET_COLUMN`
- `TRAINING_FEATURE_GROUP_NAME`
- `TRAINING_FEATURE_GROUP_VERSION`

Command-line argument:

```text
--use-cache
```

`--use-cache` loads training data from the local Parquet cache instead of Hopsworks.

## Outputs

The pipeline writes one local report artifact:

```text
reports/model_comparison_results.csv
```

The report contains one row per model candidate with:

* model name
* status
* error message, if any
* fit time
* number of features
* validation metrics
* test metrics

This artifact is a local report and is not committed to Git.

## Key Functions

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

Splits the dataset chronologically into:

* training set
* validation set
* test set

The same split is used for all model candidates.

### `split_features_and_target()`

Splits a dataframe into:

```text
X = feature columns
y = target column
```

### `build_model_candidates()`

Builds the model candidates used in the comparison.

Current candidates:

```text
HistGradientBoostingRegressor
RandomForestRegressor
XGBRegressor
```

If XGBoost is not installed, `XGBRegressor` is added as a skipped candidate instead of failing the whole pipeline.

### `compute_regression_metrics()`

Computes regression metrics with a prefix such as `valid` or `test`.

Metrics:

```text
MAE
RMSE
R2
```

### `evaluate_model_candidate()`

Trains and evaluates one model candidate.

For successful models, it returns:

* validation metrics
* test metrics
* fit time
* number of features
* status `ok`

If a model package is missing, the result status is `skipped`.

If model training or prediction fails, the result status is `failed`, and the error message is stored in the report.

### `run_model_comparison()`

Runs all model candidates and combines their results into one dataframe.

The results are sorted by:

1. model status
2. test MAE

The status ranking ensures that successful models appear before skipped or failed models.

### `save_model_comparison_results()`

Saves the comparison dataframe to:

```text
reports/model_comparison_results.csv
```

### `main()`

Runs the complete model comparison workflow:

1. load training data
2. create chronological split
3. train and evaluate all model candidates
4. print comparison summary
5. save results locally

## Model Candidates

### `HistGradientBoostingRegressor`

This is the main scikit-learn model family used in the project.

It works well for tabular weather features such as lag features, rolling aggregates and calendar features.

### `RandomForestRegressor`

This is used as a robust tree-based benchmark.

It provides a comparison against another well-known ensemble method.

### `XGBRegressor`

This adds a gradient-boosted tree model from XGBoost.

It is included as an additional model-family comparison. If the package is not installed, the candidate is skipped gracefully.

## Evaluation and Metrics

All model candidates are evaluated on the same chronological split.

The pipeline reports:

```text
valid_mae
valid_rmse
valid_r2
test_mae
test_rmse
test_r2
fit_seconds
```

MAE is the most important metric because it directly represents average temperature error in degrees Celsius.

Fit time is included to compare model cost and complexity.

## Key Design Decisions

### Same Split for All Models

All candidates are trained and evaluated on the same train/validation/test split.

This makes the comparison fairer because each model sees the same data.

### Report-Only Pipeline

This pipeline does not register or promote a model.

It only writes a local comparison report. Final model choice remains handled by the final training pipeline.

### Graceful Candidate Failures

A single missing package or failing model should not crash the entire comparison pipeline.

Each candidate gets a status:

```text
ok
skipped
failed
```

This makes the report robust and transparent.

### Explicit Status Sorting

Results are sorted so that successful models come first.

This avoids a skipped or failed model with missing metrics being sorted above valid results.

### Local Cache Mode

`--use-cache` allows model comparison without reading from Hopsworks.

This supports fast local experimentation and avoids blocking the comparison on temporary Hopsworks read issues.

## README-Relevant Points

* `model_comparison_pipeline.py` compares multiple regression model families.
* It evaluates `HistGradientBoostingRegressor`, `RandomForestRegressor` and `XGBRegressor`.
* All models use the same chronological train/validation/test split.
* Results are saved to `reports/model_comparison_results.csv`.
* The pipeline is report-only and does not upload models to Hopsworks.
* Candidate failures are handled gracefully with `ok`, `skipped` or `failed` status.
* Successful models are sorted by test MAE.

## Limitations / Future Work

* The model candidates use fixed parameters and are not independently tuned.
* The comparison uses the full `FEATURE_COLUMNS`, not the final selected feature subset.
* The comparison is based on one chronological split.
* More robust model-family comparison could use walk-forward evaluation.
* Future versions could tune each model family separately.
* Additional model families such as LightGBM or CatBoost could be added.
