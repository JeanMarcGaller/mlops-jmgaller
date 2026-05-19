# dashboard/app.py

## Role in the Project

`dashboard/app.py` provides a local Streamlit dashboard for the MLOps temperature forecasting project.

The dashboard visualizes report artifacts created by the pipeline scripts. It is a lightweight reporting layer and does not run model training, inference, monitoring, Hopsworks calls or Open-Meteo calls itself.

It helps inspect the current project state through a browser-based interface.

## Inputs

The dashboard reads local files from the `reports/` directory:

```text
reports/latest_prediction.json
reports/prediction_log.csv
reports/monitoring_results.csv
reports/backtesting_results.csv
reports/permutation_importance.csv
reports/selected_features.json
reports/model_comparison_results.csv
```

The dashboard does not call:

* Hopsworks
* Open-Meteo
* the model registry
* the feature store

It only displays existing local artifacts.

## Outputs

The dashboard does not write new artifacts.

Its output is an interactive local Streamlit web application.

It can be started with:

```bash
uv run streamlit run dashboard/app.py
```

or through the Makefile:

```bash
make dashboard
```

## Key Functions

### `read_json()`

Reads a JSON artifact if the file exists.

If the file does not exist, it returns `None`.

This allows dashboard sections to handle missing reports gracefully.

### `read_csv()`

Reads a CSV artifact if the file exists.

If the file does not exist, it returns an empty dataframe.

This avoids hard failures when a pipeline has not been run yet.

### `show_missing_artifact_message()`

Displays a consistent Streamlit info message when a report artifact is missing.

This makes the dashboard usable even when only some pipelines have already produced reports.

### `render_latest_prediction()`

Displays the latest prediction from:

```text
reports/latest_prediction.json
```

It shows key metrics such as:

* predicted temperature
* location
* model version
* model test MAE

It also displays the full prediction metadata as JSON.

### `render_prediction_log()`

Displays the cumulative prediction log from:

```text
reports/prediction_log.csv
```

If timestamps and predicted temperatures are available, it also displays a line chart of predicted temperature over time.

### `render_monitoring()`

Displays monitoring results from:

```text
reports/monitoring_results.csv
```

It shows:

* number of matched predictions
* mean absolute error
* latest absolute error
* rolling MAE, if available

It also displays charts for:

* predicted vs actual temperature
* absolute error over time

### `render_backtesting()`

Displays backtesting results from:

```text
reports/backtesting_results.csv
```

It shows:

* number of backtest splits
* mean model MAE
* mean best naive baseline MAE

It also displays a chart comparing model MAE with the best naive baseline MAE per split.

### `render_feature_selection()`

Displays feature-selection artifacts:

```text
reports/selected_features.json
reports/permutation_importance.csv
```

It shows:

* number of selected features
* number of total features
* selected feature list
* permutation importance table
* top feature importance chart

### `render_model_comparison()`

Displays model comparison results from:

```text
reports/model_comparison_results.csv
```

It shows:

* best successful model
* best test MAE
* best test R2
* full comparison table
* validation vs test MAE chart
* fit-time chart

### `main()`

Creates the Streamlit layout and navigation.

The dashboard pages are:

```text
Overview
Prediction Log
Monitoring
Backtesting
Model Comparison
Feature Selection
```

The overview page combines the latest prediction and monitoring sections.

## Dashboard Sections

### Overview

Shows:

* latest prediction
* current monitoring summary

This gives a quick status view of the project.

### Prediction Log

Shows all locally logged predictions and a prediction trend chart.

### Monitoring

Shows actual-vs-predicted performance after predictions become historically evaluable.

### Backtesting

Shows walk-forward backtesting results across multiple chronological windows.

### Model Comparison

Shows comparison results across different regression model families.

### Feature Selection

Shows selected features and permutation importance.

## Key Design Decisions

### Local Report-Based Dashboard

The dashboard reads only local report artifacts.

This keeps it simple, fast and safe to run without Hopsworks credentials or API calls.

### No Live Side Effects

The dashboard does not trigger pipelines or write data.

It is purely a visualization layer.

### Graceful Missing Artifact Handling

Not all reports exist at every stage of the project.

The dashboard handles missing files gracefully and tells the user which artifact is not available yet.

### Separate Pages

The dashboard uses sidebar navigation to separate prediction, monitoring, backtesting, model comparison and feature-selection views.

This keeps the interface understandable even as more reports are added.

## README-Relevant Points

* `dashboard/app.py` is a local Streamlit dashboard.
* It visualizes pipeline report artifacts from `reports/`.
* It does not call Hopsworks or Open-Meteo.
* It shows latest prediction, prediction log, monitoring, backtesting, model comparison and feature-selection results.
* It handles missing artifacts gracefully.
* It can be started with `uv run streamlit run dashboard/app.py` or `make dashboard`.

## Limitations / Future Work

* The dashboard is local-only.
* It reads static report artifacts and does not refresh data from external systems.
* It does not trigger pipeline runs.
* It has only basic filtering and no export functionality.
* A future version could add date filters, model-version filters and richer monitoring charts.
* A production dashboard could connect to a database or monitoring backend instead of local CSV files.
