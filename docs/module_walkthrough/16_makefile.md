# Makefile

## Role in the Project

The `Makefile` provides a central command interface for development, testing, quality checks, pipeline execution, local experiments, dashboard startup and Docker workflows.

It makes common project tasks reproducible and easier to run without remembering long `uv run ...` commands.

## Command Groups

The Makefile groups commands into several areas:

- dependency management
- testing and coverage
- code quality
- smoke tests
- feature pipeline
- training and tuning
- feature selection
- final training
- model comparison
- backtesting
- inference and monitoring
- dashboard
- end-to-end workflows
- Docker workflows

All Python commands are executed through `uv`.

## Dependency Command

### `make sync`

Installs project dependencies with:

```bash
uv sync
```

This uses the dependency definitions from `pyproject.toml` and the lockfile.

## Quality Commands

### `make test`

Runs the full test suite:

```bash
uv run pytest
```

### `make coverage`

Runs tests with terminal coverage output:

```bash
uv run pytest --cov=src --cov-report=term-missing
```

### `make coverage-html`

Runs tests and also creates an HTML coverage report:

```bash
uv run pytest --cov=src --cov-report=term-missing --cov-report=html
```

The HTML report is written to `htmlcov/`.

### `make lint`

Runs Ruff linting:

```bash
uv run ruff check .
```

### `make format`

Formats the code with Ruff:

```bash
uv run ruff format .
```

### `make quality`

Runs the main local quality gate:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

This is the recommended command before committing or pushing code.

### `make hooks`

Runs all pre-commit hooks on all files:

```bash
uv run pre-commit run --all-files
```

## Smoke Test Commands

### `make weather`

Runs the Open-Meteo API smoke test:

```bash
uv run python src/weather_api.py
```

This checks whether weather data can be fetched and converted into the expected dataframe format.

### `make hopsworks`

Runs the Hopsworks connection smoke test:

```bash
uv run python src/hopsworks_client.py
```

This checks whether the local `.env` configuration can authenticate against Hopsworks.

## Feature Pipeline Commands

### `make feature-dry`

Builds features locally for 30 days but does not write to Hopsworks:

```bash
uv run python src/feature_pipeline.py --days 30 --dry-run
```

This is useful for validating ingestion and feature engineering.

### `make feature`

Builds features for 365 historical days and writes them to Hopsworks:

```bash
uv run python src/feature_pipeline.py --days 365
```

This also creates the local feature cache.

## Training Commands

### `make train`

Runs the baseline training pipeline with Hopsworks reads and registry upload:

```bash
uv run python src/baseline_training_pipeline.py
```

### `make train-cache`

Runs baseline training from the local feature cache:

```bash
uv run python src/baseline_training_pipeline.py --use-cache
```

### `make train-local`

Runs baseline training from cache and skips the Model Registry upload:

```bash
uv run python src/baseline_training_pipeline.py --use-cache --skip-upload
```

This is useful for local experiments.

## Tuning Commands

### `make tune-fast`

Runs a small Optuna tuning job with 5 trials:

```bash
uv run python src/tuning_pipeline.py --trials 5
```

### `make tune-fast-cache`

Runs a small Optuna tuning job from the local cache:

```bash
uv run python src/tuning_pipeline.py --trials 5 --use-cache
```

### `make tune`

Runs Optuna tuning with 30 trials:

```bash
uv run python src/tuning_pipeline.py --trials 30
```

### `make tune-cache`

Runs Optuna tuning with 30 trials from the local cache:

```bash
uv run python src/tuning_pipeline.py --trials 30 --use-cache
```

## Feature Selection Commands

### `make select`

Runs feature selection with Hopsworks data access:

```bash
uv run python src/feature_selection_pipeline.py
```

### `make select-cache`

Runs feature selection from the local feature cache:

```bash
uv run python src/feature_selection_pipeline.py --use-cache
```

## Final Training Commands

### `make final`

Runs final model training and uploads the model to the Hopsworks Model Registry:

```bash
uv run python src/final_training_pipeline.py
```

### `make final-cache`

Runs final training from local cache:

```bash
uv run python src/final_training_pipeline.py --use-cache
```

### `make final-local`

Runs final training from cache and skips registry upload:

```bash
uv run python src/final_training_pipeline.py --use-cache --skip-upload
```

## Evaluation and Reporting Commands

### `make compare-models`

Runs model comparison from the local feature cache:

```bash
uv run python src/model_comparison_pipeline.py --use-cache
```

This compares:

* `HistGradientBoostingRegressor`
* `RandomForestRegressor`
* `XGBRegressor`

### `make backtest`

Runs local expanding walk-forward backtesting:

```bash
uv run python src/backtesting_pipeline.py
```

This uses the local feature cache and writes:

```text
reports/backtesting_results.csv
```

## Inference and Monitoring Commands

### `make infer`

Runs the live inference pipeline:

```bash
uv run python src/inference_pipeline.py
```

This loads the final model from Hopsworks, builds live features, writes inference features to Hopsworks and saves local prediction artifacts.

### `make monitor`

Runs the monitoring pipeline:

```bash
uv run python src/monitoring_pipeline.py
```

This compares logged predictions with actual historical temperatures once they become available.

## Dashboard Command

### `make dashboard`

Starts the local Streamlit dashboard:

```bash
uv run python -m streamlit run dashboard/app.py
```

The dashboard visualizes local report artifacts from `reports/`.

## End-to-End Commands

### `make e2e`

Runs the full Hopsworks-based workflow:

```bash
uv run python src/feature_pipeline.py --days 365
uv run python src/baseline_training_pipeline.py
uv run python src/tuning_pipeline.py --trials 30
uv run python src/feature_selection_pipeline.py
uv run python src/final_training_pipeline.py
uv run python src/inference_pipeline.py
```

This is the main complete pipeline flow.

### `make e2e-local`

Runs a local experiment workflow using the feature cache and skipping model upload:

```bash
uv run python src/tuning_pipeline.py --trials 5 --use-cache
uv run python src/feature_selection_pipeline.py --use-cache
uv run python src/final_training_pipeline.py --use-cache --skip-upload
uv run python src/model_comparison_pipeline.py --use-cache
uv run python src/backtesting_pipeline.py
```

This is useful for quick local validation without writing new model artifacts to Hopsworks.

## Docker Commands

### `make docker-build`

Builds the Docker image for `linux/amd64`:

```bash
docker build --platform linux/amd64 -t mlops-temperature-forecasting .
```

The explicit platform is useful on Apple Silicon Macs because some dependencies may not provide Linux ARM64 wheels.

### `make docker-infer`

Runs the inference pipeline inside Docker:

```bash
docker run --rm --platform linux/amd64 --env-file .env mlops-temperature-forecasting
```

### `make docker-feature-dry`

Runs a feature dry-run inside Docker:

```bash
docker run --rm --platform linux/amd64 --env-file .env mlops-temperature-forecasting \
  uv run python src/feature_pipeline.py --days 30 --dry-run
```

## Key Design Decisions

### One Command Interface

The Makefile provides a single entry point for common project operations.

This reduces command duplication in documentation and makes the workflow easier to reproduce.

### `uv` Everywhere

All Python commands are executed through `uv`.

This keeps dependency resolution and execution consistent with the project environment.

### Separate Local and Hopsworks Workflows

Many commands have cache-based local variants.

Examples:

```text
train-cache
tune-cache
select-cache
final-cache
final-local
e2e-local
```

This makes the project usable even when Hopsworks reads are slow or temporarily unavailable.

### Safe Local Training Options

Commands such as `train-local` and `final-local` skip registry upload.

This allows experimentation without polluting the Hopsworks Model Registry.

### Explicit Docker Platform

Docker commands use `linux/amd64`.

This avoids compatibility issues with dependencies on Apple Silicon machines.

## README-Relevant Points

* The Makefile provides reproducible commands for development and pipeline execution.
* `make quality` is the main local quality gate.
* `make feature`, `make train`, `make tune`, `make select`, `make final`, `make infer` run the main pipeline stages.
* `make e2e` runs the complete Hopsworks workflow.
* `make e2e-local` runs a local cache-based experiment workflow.
* `make dashboard` starts the local Streamlit dashboard.
* Docker commands are available for containerized execution.
* Local cache variants make development robust against temporary Hopsworks read issues.

## Limitations / Future Work

* The Makefile does not include parameters for every pipeline option.
* Some commands use fixed defaults, such as 365 feature days or 30 tuning trials.
* More configurable Make targets could be added with environment variables.
* The Makefile does not orchestrate scheduled inference or monitoring.
* A future version could add deployment or release commands.
