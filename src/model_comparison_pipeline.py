"""Model comparison pipeline for the temperature forecasting project.

This pipeline compares multiple regression models on the same chronological
train/validation/test split.

It is intentionally report-based and local/Hopsworks-read compatible:
- input: Hopsworks Training Feature Group or local feature cache
- output: reports/model_comparison_results.csv
"""

import argparse
import time

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    FEATURE_CACHE_PATH,
    MODEL_COMPARISON_RESULTS_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
    TRAINING_FEATURE_GROUP_NAME,
    TRAINING_FEATURE_GROUP_VERSION,
)
from features import FEATURE_COLUMNS
from hopsworks_client import get_feature_store
from hopsworks_read_utils import (
    read_dataframe_from_cache,
    read_feature_group_with_retry,
)

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None


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
        raise ValueError("Missing event_time column for time-based split.")

    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time").reset_index(drop=True)

    return df


def load_training_dataframe(
    fs,
    use_cache: bool = False,
) -> pd.DataFrame:
    """Load training data from Hopsworks or local cache."""
    if use_cache:
        print("Loading training data from local cache because --use-cache is set.")

        df = read_dataframe_from_cache(FEATURE_CACHE_PATH)

        return validate_training_dataframe(df)

    print("Loading training data from Hopsworks Feature Group with robust read...")

    feature_group = fs.get_feature_group(
        name=TRAINING_FEATURE_GROUP_NAME,
        version=TRAINING_FEATURE_GROUP_VERSION,
    )

    return read_feature_group_with_retry(
        feature_group=feature_group,
        cache_path=FEATURE_CACHE_PATH,
        validate_fn=validate_training_dataframe,
        context="model comparison",
    )


def time_based_train_valid_test_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    valid_fraction: float = 0.15,
):
    """Split data chronologically into train, validation and test sets."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1.")

    if train_fraction + valid_fraction >= 1:
        raise ValueError("train_fraction + valid_fraction must be smaller than 1.")

    n_rows = len(df)

    train_end = int(n_rows * train_fraction)
    valid_end = int(n_rows * (train_fraction + valid_fraction))

    if train_end <= 0 or valid_end <= train_end or valid_end >= n_rows:
        raise ValueError("Dataset is too small for the requested split fractions.")

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()

    print()
    print("Time-based split")
    print("----------------")
    print(f"train rows: {len(train_df)}")
    print(f"valid rows: {len(valid_df)}")
    print(f"test rows:  {len(test_df)}")
    print(f"train_until: {train_df['event_time'].max()}")
    print(f"valid_from:  {valid_df['event_time'].min()}")
    print(f"valid_until: {valid_df['event_time'].max()}")
    print(f"test_from:   {test_df['event_time'].min()}")

    return train_df, valid_df, test_df


def split_features_and_target(
    df: pd.DataFrame,
    feature_columns: list[str],
):
    """Split dataframe into model features and target."""
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def build_model_candidates() -> list[tuple[str, object | None]]:
    """Build model candidates for comparison."""
    candidates: list[tuple[str, object | None]] = [
        (
            "HistGradientBoostingRegressor",
            HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=300,
                learning_rate=0.06,
                max_leaf_nodes=31,
                min_samples_leaf=30,
                l2_regularization=0.1,
                random_state=42,
                early_stopping=False,
            ),
        ),
        (
            "RandomForestRegressor",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    if XGBRegressor is None:
        candidates.append(("XGBRegressor", None))
    else:
        candidates.append(
            (
                "XGBRegressor",
                XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.04,
                    max_depth=5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        )

    return candidates


def compute_regression_metrics(y_true, y_pred, prefix: str) -> dict:
    """Compute regression metrics with a metric-name prefix."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return {
        f"{prefix}_mae": float(mae),
        f"{prefix}_rmse": float(rmse),
        f"{prefix}_r2": float(r2),
    }


def evaluate_model_candidate(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Train and evaluate one model candidate."""
    if model is None:
        return {
            "model_name": model_name,
            "status": "skipped",
            "error_message": "Model package is not installed.",
            "fit_seconds": None,
            "n_features": int(X_train.shape[1]),
            "valid_mae": None,
            "valid_rmse": None,
            "valid_r2": None,
            "test_mae": None,
            "test_rmse": None,
            "test_r2": None,
        }

    start_time = time.perf_counter()

    try:
        model.fit(X_train, y_train)

        fit_seconds = time.perf_counter() - start_time

        valid_pred = model.predict(X_valid)
        test_pred = model.predict(X_test)

        result = {
            "model_name": model_name,
            "status": "ok",
            "error_message": "",
            "fit_seconds": float(fit_seconds),
            "n_features": int(X_train.shape[1]),
        }

        result.update(
            compute_regression_metrics(
                y_true=y_valid,
                y_pred=valid_pred,
                prefix="valid",
            )
        )
        result.update(
            compute_regression_metrics(
                y_true=y_test,
                y_pred=test_pred,
                prefix="test",
            )
        )

        return result

    except Exception as exc:  # pragma: no cover
        fit_seconds = time.perf_counter() - start_time

        return {
            "model_name": model_name,
            "status": "failed",
            "error_message": f"{type(exc).__name__}: {exc}",
            "fit_seconds": float(fit_seconds),
            "n_features": int(X_train.shape[1]),
            "valid_mae": None,
            "valid_rmse": None,
            "valid_r2": None,
            "test_mae": None,
            "test_rmse": None,
            "test_r2": None,
        }


def run_model_comparison(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Run all model candidates and return comparison results."""
    X_train, y_train = split_features_and_target(train_df, feature_columns)
    X_valid, y_valid = split_features_and_target(valid_df, feature_columns)
    X_test, y_test = split_features_and_target(test_df, feature_columns)

    results = []

    for model_name, model in build_model_candidates():
        print()
        print(f"Evaluating model: {model_name}")
        print("-----------------")

        result = evaluate_model_candidate(
            model_name=model_name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
        )

        results.append(result)

        if result["status"] == "ok":
            print(f"valid_mae: {result['valid_mae']:.4f}")
            print(f"test_mae:  {result['test_mae']:.4f}")
            print(f"fit_seconds: {result['fit_seconds']:.2f}")
        else:
            print(f"status: {result['status']}")
            print(f"reason: {result['error_message']}")

    comparison_df = pd.DataFrame(results)

    status_rank = {
        "ok": 0,
        "skipped": 1,
        "failed": 2,
    }

    comparison_df["status_rank"] = comparison_df["status"].map(status_rank).fillna(99)

    comparison_df = (
        comparison_df.sort_values(
            by=["status_rank", "test_mae"],
            ascending=[True, True],
            na_position="last",
        )
        .drop(columns=["status_rank"])
        .reset_index(drop=True)
    )

    return comparison_df


def save_model_comparison_results(comparison_df: pd.DataFrame) -> None:
    """Save model comparison results locally."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(MODEL_COMPARISON_RESULTS_PATH, index=False)

    print()
    print("Saved model comparison results")
    print("------------------------------")
    print(f"Model comparison results: {MODEL_COMPARISON_RESULTS_PATH}")


def main(use_cache: bool = False) -> None:
    """Run the complete model-comparison workflow."""
    fs = None

    if not use_cache:
        fs = get_feature_store()

    df = load_training_dataframe(
        fs=fs,
        use_cache=use_cache,
    )

    train_df, valid_df, test_df = time_based_train_valid_test_split(df)

    comparison_df = run_model_comparison(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        feature_columns=FEATURE_COLUMNS,
    )

    print()
    print("Model comparison summary")
    print("------------------------")
    print(comparison_df)

    save_model_comparison_results(comparison_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare multiple regression models.")

    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use local Parquet feature cache instead of reading from Hopsworks.",
    )

    args = parser.parse_args()

    main(use_cache=args.use_cache)
