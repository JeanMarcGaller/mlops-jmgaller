# src/feature_pipeline.py

## Role in the Project

`feature_pipeline.py` creates the historical training dataset for the temperature forecasting project.

It connects the first parts of the pipeline:

- `weather_api.py` loads historical raw weather data
- `features.py` creates model features and the supervised target
- `config.py` provides paths, target settings and Hopsworks object names
- `hopsworks_client.py` provides access to the Hopsworks Feature Store

The pipeline writes only historical rows with known target values. Live inference rows are handled separately by `inference_pipeline.py`.

## Inputs

The pipeline uses:

- historical weather data from Open-Meteo
- feature engineering logic from `features.py`
- configuration values from `config.py`
- optional command-line arguments

Important config values:

- `DATA_FEATURES_DIR`
- `FEATURE_CACHE_PATH`
- `FORECAST_HORIZON_HOURS`
- `TARGET_COLUMN`
- `TRAINING_FEATURE_GROUP_NAME`
- `TRAINING_FEATURE_GROUP_VERSION`

Command-line arguments:

```text
--days
--dry-run
```

`--days` controls how many historical days are fetched from Open-Meteo.

`--dry-run` builds and caches the feature dataframe locally but skips the Hopsworks write step.

## Outputs

The pipeline produces two main outputs:

1. A local Parquet feature cache

```text
data/features/temperature_features.parquet
```

2. A Hopsworks Training Feature Group

```text
temperature_forecast_training_features
version 3
```

The feature dataframe contains:

* primary key columns
* event-time column
* model features
* target column

## Key Functions

### `build_and_cache_training_features()`

Fetches historical weather data, builds training features and saves a local Parquet cache.

The function:

1. validates the requested number of days
2. defines a historical date range
3. fetches historical weather data from Open-Meteo
4. builds model features with `include_target=True`
5. writes the engineered feature dataframe to local Parquet
6. prints basic diagnostics such as shape, columns and target summary

The historical end date is set to two days before today. This buffer avoids requesting very recent archive data that may not yet be fully available from Open-Meteo.

### `write_to_feature_store()`

Writes the engineered training features to Hopsworks.

The Feature Group is created if it does not exist yet.

It uses:

```text
primary key: location_id, event_time_unix
event-time:  event_time
online:      enabled
```

The Feature Group name and version are read from `config.py`.

### `main()`

Runs the feature pipeline.

It always builds and caches the training features locally.

If `dry_run=True`, it stops before writing to Hopsworks.

If `dry_run=False`, it writes the feature dataframe to the Hopsworks Training Feature Group.

## Key Design Decisions

### Historical Rows Only

This pipeline writes only historical rows where the future target is known.

The target is created by `features.py` with `include_target=True`.

Live inference rows are not written here. They are handled by `inference_pipeline.py` and stored in a separate Inference Feature Group.

### Local Feature Cache

The pipeline always writes a local Parquet cache.

This cache is used by offline workflows such as:

* local training experiments
* tuning with `--use-cache`
* feature selection with `--use-cache`
* final training with `--use-cache`
* model comparison
* walk-forward backtesting
* fallback when Hopsworks offline reads fail

This makes the project more robust during development.

### Separate Training Feature Group

Historical training features are written to a dedicated Training Feature Group.

This keeps historical rows with known labels separate from live inference rows where the future target is not known.

### Two-Day Historical Buffer

The pipeline fetches data only until two days before today.

This avoids relying on very recent historical archive data that may not yet be fully complete or available.

### Dry-Run Mode

`--dry-run` allows validating the data ingestion and feature engineering process without writing to Hopsworks.

This is useful for development, debugging and local checks.

## Validation and Error Handling

The pipeline validates that `days` is greater than 0.

Further validation happens in downstream modules:

* `weather_api.py` validates Open-Meteo responses
* `features.py` validates raw dataframe schema and unique timestamp keys
* Hopsworks write errors surface during `feature_group.insert()`

## Point-in-Time / Leakage Considerations

The feature pipeline uses historical data and creates the target only for timestamps where the future temperature is known.

The actual leakage prevention is implemented in `features.py`:

* exact timestamp-based lag features
* exact timestamp-based target generation
* time-based rolling windows
* current row excluded from rolling windows

This pipeline keeps training data separate from live inference data by writing to the Training Feature Group only.

## Limitations / Future Work

* The pipeline currently fetches historical actual weather data, not historical forecast snapshots.
* A production-grade forecasting setup could train on historical forecast snapshots with forecast issue times to reduce training-serving skew.
* The local cache does not currently store metadata such as creation date, source date range or Feature Group version.
* The pipeline currently supports one configured location at a time.
* Hopsworks writes are synchronous with `wait=True`, which is simple and safe but can take time.
