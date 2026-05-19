"""Utilities for robust Hopsworks Feature Store reads.

Hopsworks offline reads can occasionally fail due to temporary Feature Query
Service / Arrow Flight issues. This module centralizes retry logic and local
Parquet cache fallback behavior for offline pipelines.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_READ_MAX_ATTEMPTS = 3
DEFAULT_READ_RETRY_WAIT_SECONDS = 120
DEFAULT_ARROW_FLIGHT_TIMEOUT_SECONDS = 900


def read_feature_group_with_retry(
    feature_group: Any,
    cache_path: Path,
    validate_fn: Callable[[pd.DataFrame], pd.DataFrame],
    context: str,
    max_attempts: int = DEFAULT_READ_MAX_ATTEMPTS,
    wait_seconds: int = DEFAULT_READ_RETRY_WAIT_SECONDS,
    use_hive: bool = True,
    arrow_flight_timeout_seconds: int = DEFAULT_ARROW_FLIGHT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Read a Hopsworks Feature Group with retry and local cache fallback.

    Args:
        feature_group: Hopsworks Feature Group object.
        cache_path: Local Parquet cache path used as fallback.
        validate_fn: Function used to validate and sort the loaded dataframe.
        context: Human-readable pipeline context for logs and error messages.
        max_attempts: Maximum number of Hopsworks read attempts.
        wait_seconds: Seconds to wait between failed attempts.
        use_hive: Whether to request Hive-based reading from Hopsworks.
        arrow_flight_timeout_seconds: Arrow Flight timeout in seconds.

    Returns:
        Validated dataframe from Hopsworks or local cache.

    Raises:
        FileNotFoundError: If Hopsworks read fails and local cache is missing.
        ValueError: If max_attempts or wait_seconds are invalid.
    """
    if max_attempts <= 0:
        raise ValueError("max_attempts must be greater than 0.")

    if wait_seconds < 0:
        raise ValueError("wait_seconds must be greater than or equal to 0.")

    last_exception = None
    df = None

    read_options = {
        "use_hive": use_hive,
        "arrow_flight_config": {
            "timeout": arrow_flight_timeout_seconds,
        },
    }

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Read attempt {attempt}/{max_attempts}...")
            print(f"Read context: {context}")

            df = feature_group.read(read_options=read_options)

            print("Loaded data from Hopsworks.")
            print(f"Raw dataframe shape: {df.shape}")
            break

        except Exception as exc:
            last_exception = exc

            print(f"Read attempt {attempt} failed.")
            print(f"Reason: {type(exc).__name__}: {exc}")

            if attempt < max_attempts:
                print(f"Waiting {wait_seconds} seconds before retry...")
                time.sleep(wait_seconds)

    if df is None:
        print()
        print(f"Could not read data from Hopsworks for: {context}")
        print("Using local feature cache fallback.")
        print(
            f"Last Hopsworks error: {type(last_exception).__name__}: {last_exception}"
        )

        df = read_dataframe_from_cache(
            cache_path=cache_path,
            last_exception=last_exception,
        )

    return validate_fn(df)


def read_dataframe_from_cache(
    cache_path: Path,
    last_exception: Exception | None = None,
) -> pd.DataFrame:
    """Read a dataframe from a local Parquet cache.

    Args:
        cache_path: Local Parquet cache path.
        last_exception: Optional original exception used as exception cause.

    Returns:
        Cached dataframe.

    Raises:
        FileNotFoundError: If the cache does not exist.
    """
    if not cache_path.exists():
        message = (
            f"Local feature cache not found at {cache_path}. "
            "Run `uv run python src/feature_pipeline.py --days 365` first."
        )

        if last_exception is not None:
            raise FileNotFoundError(message) from last_exception

        raise FileNotFoundError(message)

    print()
    print("Loading data from local feature cache fallback...")
    print(f"Cache path: {cache_path}")

    df = pd.read_parquet(cache_path)

    print("Loaded data from local feature cache.")
    print(f"Raw dataframe shape: {df.shape}")

    return df
