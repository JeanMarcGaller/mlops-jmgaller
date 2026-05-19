"""Backtesting pipeline for the temperature forecasting project.

This pipeline evaluates model performance across multiple chronological
train/validation/test windows.

The goal is to check whether model improvements are stable over time instead
of relying on a single train/validation/test split.
"""

import argparse

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    BACKTESTING_RESULTS_PATH,
    FEATURE_CACHE_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from features import FEATURE_COLUMNS
from hopsworks_read_utils import read_dataframe_from_cache

DEFAULT_N_SPLITS = 4
DEFAULT_INITIAL_TRAIN_FRACTION = 0.50
DEFAULT_VALID_FRACTION = 0.10
DEFAULT_TEST_FRACTION = 0.10


def add_fixed_model_params(params: dict | None = None) -> dict:
    """Add fixed HistGradientBoostingRegressor parameters."""
    if params is None:
        params = {}

    return {
        **params,
        "loss": "squared_error",
        "max_iter": 200,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "early_stopping": False,
        "random_state": 42,
    }


def validate_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and chronologically sort the training dataframe."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    missing_features = [
        feature for feature in FEATURE_COLUMNS if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    if "event_time" not in df.columns:
        raise ValueError("Missing event_time column for time-based backtesting.")

    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time").reset_index(drop=True)

    return df


def load_training_dataframe(use_cache: bool = True) -> pd.DataFrame:
    """Load training data for backtesting.

    Backtesting starts as a local/offline workflow. It currently reads the
    feature cache created by feature_pipeline.py.
    """
    if not use_cache:
        raise NotImplementedError(
            "Backtesting currently supports only --use-cache. "
            "Run feature_pipeline.py first to create the local feature cache."
        )

    print("Loading backtesting data from local feature cache...")

    df = read_dataframe_from_cache(FEATURE_CACHE_PATH)

    return validate_training_dataframe(df)


def split_features_and_target(
    df: pd.DataFrame,
    feature_columns: list[str],
):
    """Split dataframe into model features and target."""
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def create_backtest_windows(
    df: pd.DataFrame,
    n_splits: int = DEFAULT_N_SPLITS,
    initial_train_fraction: float = DEFAULT_INITIAL_TRAIN_FRACTION,
    valid_fraction: float = DEFAULT_VALID_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Create expanding-window walk-forward backtest windows.

    Each split trains on all data available up to that point, validates on the
    immediately following validation window and tests on the next test window.

    This simulates repeated production retraining over time:

    - split 1: train on early history, validate, test on next period
    - split 2: train on more history, validate, test on next period
    - split N: train on all history available before that split's validation
      and test period
    """
    if n_splits <= 0:
        raise ValueError("n_splits must be greater than 0.")

    if not 0 < initial_train_fraction < 1:
        raise ValueError("initial_train_fraction must be between 0 and 1.")

    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1.")

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    n_rows = len(df)

    initial_train_size = int(n_rows * initial_train_fraction)
    valid_size = int(n_rows * valid_fraction)
    test_size = int(n_rows * test_fraction)

    if initial_train_size <= 0 or valid_size <= 0 or test_size <= 0:
        raise ValueError("Dataset is too small for the requested backtest fractions.")

    required_rows = initial_train_size + valid_size + (n_splits * test_size)

    if required_rows > n_rows:
        raise ValueError(
            "Dataset is too small for the requested walk-forward setup. "
            f"Required rows: {required_rows}, available rows: {n_rows}. "
            "Reduce --splits, initial_train_fraction, valid_fraction or "
            "test_fraction."
        )

    windows = []

    for split_index in range(n_splits):
        train_end = initial_train_size + (split_index * test_size)
        valid_end = train_end + valid_size
        test_end = valid_end + test_size

        train_df = df.iloc[:train_end].copy()
        valid_df = df.iloc[train_end:valid_end].copy()
        test_df = df.iloc[valid_end:test_end].copy()

        if train_df.empty or valid_df.empty or test_df.empty:
            raise ValueError("One of the backtest windows has an empty split.")

        windows.append((train_df, valid_df, test_df))

    return windows


def evaluate_naive_baselines(test_df: pd.DataFrame) -> dict:
    """Evaluate persistence and seasonal naive baselines."""
    y_test = test_df[TARGET_COLUMN]

    baseline_predictions = {
        "persistence": test_df["temperature_2m"],
        "seasonal_naive_24h": test_df["temperature_lag_24h"],
    }

    metrics = {}

    for baseline_name, y_pred in baseline_predictions.items():
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        metrics[f"{baseline_name}_mae"] = float(mae)
        metrics[f"{baseline_name}_rmse"] = float(rmse)
        metrics[f"{baseline_name}_r2"] = float(r2)

    metrics["best_naive_baseline_mae"] = min(
        metrics["persistence_mae"],
        metrics["seasonal_naive_24h_mae"],
    )

    return metrics


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """Train one baseline model for a backtest window."""
    model = HistGradientBoostingRegressor(**add_fixed_model_params())
    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate model on a backtest test split."""
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    return {
        "model_mae": float(mae),
        "model_rmse": float(rmse),
        "model_r2": float(r2),
    }


def evaluate_backtest_window(
    split_id: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Train and evaluate one backtest window."""
    train_valid_df = pd.concat(
        [train_df, valid_df],
        axis=0,
        ignore_index=True,
    )

    X_train_valid, y_train_valid = split_features_and_target(
        train_valid_df,
        FEATURE_COLUMNS,
    )
    X_test, y_test = split_features_and_target(
        test_df,
        FEATURE_COLUMNS,
    )

    model = train_model(
        X_train=X_train_valid,
        y_train=y_train_valid,
    )

    baseline_metrics = evaluate_naive_baselines(test_df)
    model_metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    best_naive_mae = baseline_metrics["best_naive_baseline_mae"]
    model_mae = model_metrics["model_mae"]

    if best_naive_mae > 0:
        improvement = (best_naive_mae - model_mae) / best_naive_mae
    else:
        improvement = 0.0

    return {
        "split_id": split_id,
        "train_start": train_df["event_time"].min(),
        "train_end": train_df["event_time"].max(),
        "valid_start": valid_df["event_time"].min(),
        "valid_end": valid_df["event_time"].max(),
        "test_start": test_df["event_time"].min(),
        "test_end": test_df["event_time"].max(),
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_test": len(test_df),
        **baseline_metrics,
        **model_metrics,
        "mae_improvement_over_best_naive_baseline": float(improvement),
    }


def run_backtesting(
    df: pd.DataFrame,
    n_splits: int = DEFAULT_N_SPLITS,
) -> pd.DataFrame:
    """Run expanding-window walk-forward backtesting."""
    windows = create_backtest_windows(
        df=df,
        n_splits=n_splits,
    )

    rows = []

    for split_index, (train_df, valid_df, test_df) in enumerate(windows, start=1):
        print()
        print(f"Walk-forward {split_index}/{len(windows)}")
        print("--------------------------------")
        print(
            f"train: {train_df['event_time'].min()} -> {train_df['event_time'].max()}"
        )
        print(
            f"valid: {valid_df['event_time'].min()} -> {valid_df['event_time'].max()}"
        )
        print(f"test:  {test_df['event_time'].min()} -> {test_df['event_time'].max()}")

        row = evaluate_backtest_window(
            split_id=split_index,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
        )

        print(f"model_mae: {row['model_mae']:.4f}")
        print(f"best_naive_baseline_mae: {row['best_naive_baseline_mae']:.4f}")
        print(
            "mae_improvement_over_best_naive_baseline: "
            f"{row['mae_improvement_over_best_naive_baseline']:.2%}"
        )

        rows.append(row)

    return pd.DataFrame(rows)


def save_backtesting_results(results_df: pd.DataFrame) -> None:
    """Save backtesting results locally."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(BACKTESTING_RESULTS_PATH, index=False)

    print()
    print("Saved backtesting results")
    print("-------------------------")
    print(f"Backtesting results: {BACKTESTING_RESULTS_PATH}")


def main(
    use_cache: bool = True,
    n_splits: int = DEFAULT_N_SPLITS,
) -> None:
    """Run the complete backtesting workflow."""
    df = load_training_dataframe(use_cache=use_cache)

    results_df = run_backtesting(
        df=df,
        n_splits=n_splits,
    )

    print()
    print("Backtesting summary")
    print("-------------------")
    print(results_df[["split_id", "model_mae", "best_naive_baseline_mae"]])
    print(f"Mean model MAE: {results_df['model_mae'].mean():.4f}")
    print(f"Mean best naive MAE: {results_df['best_naive_baseline_mae'].mean():.4f}")
    print(
        "Mean MAE improvement over best naive baseline: "
        f"{results_df['mae_improvement_over_best_naive_baseline'].mean():.2%}"
    )

    save_backtesting_results(results_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run chronological backtesting for the temperature model."
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Disable local Parquet feature cache. Currently unsupported for "
            "backtesting."
        ),
    )

    parser.add_argument(
        "--splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help="Number of walk-forward backtest splits.",
    )

    args = parser.parse_args()

    main(
        use_cache=not args.no_cache,
        n_splits=args.splits,
    )
