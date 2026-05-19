# src/hopsworks_read_utils.py

## Role in the Project

`hopsworks_read_utils.py` provides utilities for reading offline feature data from Hopsworks.

Some pipeline steps need to read training data from the Hopsworks Feature Store. In practice, offline reads can occasionally fail due to temporary Feature Query Service or Arrow Flight issues. This module centralizes retry behavior and provides a local Parquet cache fallback.

It is mainly used by offline training and evaluation pipelines such as:

- baseline training
- tuning
- feature selection
- final training
- model comparison

## Inputs

The module works with:

- a Hopsworks Feature Group object
- a local Parquet cache path
- a validation function for the loaded dataframe
- a human-readable context string for logging
- retry settings

Important default settings:

```text
DEFAULT_READ_MAX_ATTEMPTS = 3
DEFAULT_READ_RETRY_WAIT_SECONDS = 120
DEFAULT_ARROW_FLIGHT_TIMEOUT_SECONDS = 900
```

## Outputs

The module returns a validated pandas dataframe.

The dataframe can come from:

1. Hopsworks Feature Store
2. local Parquet feature cache fallback

In both cases, the returned dataframe is passed through the provided validation function before it is returned to the pipeline.

## Key Functions

### `read_feature_group_with_retry()`

Reads a Hopsworks Feature Group with retry logic and local cache fallback.

The function tries to read from Hopsworks up to `max_attempts` times.

For each attempt, it logs:

* attempt number
* pipeline context
* success or failure reason
* raw dataframe shape on success

If all Hopsworks read attempts fail, it falls back to the local Parquet cache.

The loaded dataframe is then passed through `validate_fn`.

This makes the caller responsible for defining what a valid dataframe means for a specific pipeline.

### `read_dataframe_from_cache()`

Reads a dataframe from a local Parquet cache.

If the cache file does not exist, the function raises a `FileNotFoundError` with a clear message explaining how to create the cache.

The expected cache is usually:

```text
data/features/temperature_features.parquet
```

created by:

```bash
uv run python src/feature_pipeline.py --days 365
```

## Key Design Decisions

### Centralized Retry Logic

Retry handling is implemented in one shared utility instead of being duplicated across training, tuning, feature selection and evaluation pipelines.

This keeps the pipelines simpler and makes read behavior consistent.

### Local Cache Fallback

If Hopsworks offline reads fail, the module can fall back to a local Parquet cache.

This is important for local development and experimentation because temporary Hopsworks read issues should not block every offline ML workflow.

### Validation Function Injection

The module receives a `validate_fn` from the caller.

This keeps the read utility generic. Different pipelines can apply their own validation and sorting logic after loading the dataframe.

### Explicit Read Options

The Hopsworks read uses explicit read options:

```python
read_options = {
    "use_hive": use_hive,
    "arrow_flight_config": {
        "timeout": arrow_flight_timeout_seconds,
    },
}
```

This makes the Hopsworks read behavior visible and configurable.

## Validation and Error Handling

The module validates:

* `max_attempts` must be greater than 0
* `wait_seconds` must be greater than or equal to 0

It handles failed Hopsworks reads by:

1. storing the last exception
2. retrying after a configurable wait time
3. falling back to the local cache if all attempts fail

If the local cache is missing, it raises a `FileNotFoundError`.

If the original Hopsworks error is available, it is attached as the exception cause.

## Limitations / Future Work

* The retry wait time is currently static.
* A future version could use exponential backoff.
* The current fallback assumes that the local cache is recent enough for the experiment.
* A future version could store metadata about cache creation time, Feature Group version or source date range.
* The module currently prints logs directly; a production version could use structured logging.
