# src/config.py

## Role in the Project

`config.py` is the central configuration file for the project. All pipeline scripts import shared paths, Hopsworks settings, Feature Store object names, model names, location settings and forecasting parameters from here.

## Inputs

The file loads local environment variables from `.env` using `python-dotenv`.

Important `.env` variables include:

- `HOPSWORKS_HOST`
- `HOPSWORKS_PORT`
- `HOPSWORKS_PROJECT`
- `HOPSWORKS_API_KEY`
- `HOPSWORKS_ENGINE`
- `LATITUDE`
- `LONGITUDE`
- `LOCATION_ID`
- `TIMEZONE`
- `FORECAST_HORIZON_HOURS`
- `MODEL_VERSION`

Several values have safe defaults:

- `Basel` as the default location
- `Europe/Zurich` as the timezone
- `6` hours as the forecast horizon.

## Outputs

`config.py` exposes constants for:

- local project directories
- local artifact paths
- Hopsworks connection settings
- weather location settings
- target configuration
- Feature Group names and versions
- Feature View names and versions
- Model Registry names
- optional pinned model version for inference

## Key Design Decisions

### Centralized Path Management

All local artifacts are defined through shared paths:

- `data/features/temperature_features.parquet`
- `models/model.joblib`
- `reports/*.json`
- `reports/*.csv`

This ensures that training, tuning, feature selection, inference, monitoring and the dashboard use the same artifact locations.

### Separate Training and Inference Artifacts

Training and inference use separate Feature Groups and Feature Views:

- `temperature_forecast_training_features`
- `temperature_forecast_inference_features`
- `temperature_forecast_training_fv`
- `temperature_forecast_inference_fv`

This prevents historical training rows with known targets from being mixed with live inference rows where the future target is not known yet.

### Feature Store Version 3

The current pipeline uses Feature Group and Feature View version `3`.

This version represents the updated time-based feature engineering logic:

- exact timestamp-based lag features
- exact timestamp-based target generation
- time-based rolling windows
- repaired live inference context

### Separate Model Registry Names

Baseline and final models use separate registry names:

- `temperature_forecast_baseline_regressor`
- `temperature_forecast_final_regressor`

The inference pipeline loads the final model by default. This prevents a newly uploaded baseline model from accidentally being loaded as the latest production candidate.

### Optional Model Version Pinning

`MODEL_VERSION` can be set in `.env` to load a specific model version.

If `MODEL_VERSION` is empty, the inference pipeline automatically loads the latest available version of the final model.

## Validation

`config.py` contains lightweight runtime validation for common configuration mistakes:

- forecast horizon must be greater than 0
- `LOCATION_ID` must not be empty
- `TIMEZONE` must not be empty
- latitude must be between -90 and 90
- longitude must be between -180 and 180
- `MODEL_VERSION` must be either empty or an integer

## Limitations / Future Work

- Secrets such as `HOPSWORKS_API_KEY` are intentionally not validated at import time, so local tests and offline workflows can run without Hopsworks credentials.
- More production-oriented deployments could separate configurations for `dev`, `staging` and `prod`.
- Event-time handling could be hardened in a later version by adding an explicit UTC-normalized timestamp column.