"""Baseline training pipeline for the temperature forecasting project.

This pipeline trains the first supervised ML model for predicting the
temperature several hours ahead.

It performs the following steps:

1. Create or load the Hopsworks Training Feature View.
2. Load historical training data from the Hopsworks Training Feature Group.
3. Create a chronological train/validation/test split.
4. Evaluate persistence and seasonal naive baselines.
5. Train a baseline HistGradientBoostingRegressor.
6. Evaluate the model on validation and test data.
7. Save the model locally as a joblib bundle.
8. Upload the model artifact to the Hopsworks Model Registry.

This pipeline is intentionally simple. It provides a reproducible baseline before
later steps such as Optuna tuning, feature selection and final training.
"""

import argparse
import json
import shutil

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    BASELINE_MODEL_NAME,
    FEATURE_CACHE_PATH,
    MODEL_DIR,
    MODEL_LOCAL_PATH,
    TARGET_COLUMN,
    TRAINING_FEATURE_GROUP_NAME,
    TRAINING_FEATURE_GROUP_VERSION,
    TRAINING_FEATURE_VIEW_NAME,
    TRAINING_FEATURE_VIEW_VERSION,
)
from features import FEATURE_COLUMNS
from hopsworks_client import get_feature_store, get_model_registry
from hopsworks_read_utils import (
    read_dataframe_from_cache,
    read_feature_group_with_retry,
)


def get_or_create_training_feature_view(fs):
    """Create or load the Hopsworks Training Feature View.

    The Feature View provides a reusable training interface over the Training
    Feature Group. In this project, the actual dataframe read is done directly
    from the Feature Group with `use_hive=True`, because this was more stable
    for local batch reads. The Feature View is still created to satisfy the FTI
    architecture and to keep the Hopsworks setup explicit.

    Args:
        fs: Hopsworks Feature Store object.

    Returns:
        Hopsworks Feature View object.
    """
    feature_group = fs.get_feature_group(
        name=TRAINING_FEATURE_GROUP_NAME,
        version=TRAINING_FEATURE_GROUP_VERSION,
    )

    query = feature_group.select_all()

    try:
        feature_view = fs.get_feature_view(
            name=TRAINING_FEATURE_VIEW_NAME,
            version=TRAINING_FEATURE_VIEW_VERSION,
        )

        if feature_view is not None:
            print(
                f"Using existing training feature view: "
                f"{TRAINING_FEATURE_VIEW_NAME}, version {TRAINING_FEATURE_VIEW_VERSION}"
            )
            return feature_view

    except Exception as exc:
        print(
            f"Training feature view {TRAINING_FEATURE_VIEW_NAME}, "
            f"version {TRAINING_FEATURE_VIEW_VERSION} was not found."
        )
        print(f"Reason: {exc}")

    print(
        f"Creating training feature view: "
        f"{TRAINING_FEATURE_VIEW_NAME}, version {TRAINING_FEATURE_VIEW_VERSION}"
    )

    feature_view = fs.create_feature_view(
        name=TRAINING_FEATURE_VIEW_NAME,
        version=TRAINING_FEATURE_VIEW_VERSION,
        query=query,
        labels=[TARGET_COLUMN],
        description=(
            "Training feature view for 6-hour temperature forecasting. "
            "Includes current, lag, rolling, trend and calendar features."
        ),
    )

    if feature_view is None:
        raise RuntimeError(
            "Feature view creation returned None. "
            "Increase TRAINING_FEATURE_VIEW_VERSION or check Hopsworks."
        )

    return feature_view


def validate_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and sort the training dataframe.

    This function is separated from `load_training_dataframe` so it can be unit
    tested without connecting to Hopsworks.

    Args:
        df: Raw training dataframe read from Hopsworks.

    Returns:
        Validated dataframe sorted by event_time.

    Raises:
        ValueError: If target, feature columns or event_time are missing.
    """
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
        context="baseline training",
    )


def time_based_train_valid_test_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    valid_fraction: float = 0.15,
):
    """Split data chronologically into train, validation and test sets.

    Time-series data should not be randomly split because that can leak future
    information into the training set. This function keeps the chronological
    order:

    - first part: training
    - middle part: validation
    - last part: test

    Args:
        df: Chronologically sorted dataframe.
        train_fraction: Fraction used for training.
        valid_fraction: Fraction used for validation.

    Returns:
        Tuple of train_df, valid_df and test_df.
    """
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


def split_features_and_target(df: pd.DataFrame):
    """Split a dataframe into model input features and target series."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def evaluate_naive_baselines(
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate naive forecasting baselines on the test set.

    The persistence baseline predicts that the future temperature equals the
    current temperature.

    The seasonal naive baseline predicts that the future temperature equals the
    temperature at the same feature time one day before.
    """
    baseline_predictions = {
        "persistence": X_test["temperature_2m"],
        "seasonal_naive_24h": X_test["temperature_lag_24h"],
    }

    metrics = {}

    print()
    print("Naive Baselines")
    print("---------------")

    for baseline_name, y_pred in baseline_predictions.items():
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        metrics[f"{baseline_name}_mae"] = float(mae)
        metrics[f"{baseline_name}_rmse"] = float(rmse)
        metrics[f"{baseline_name}_r2"] = float(r2)

        print(baseline_name)
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}")
        print()

    best_naive_baseline_mae = min(
        metrics["persistence_mae"],
        metrics["seasonal_naive_24h_mae"],
    )

    metrics["best_naive_baseline_mae"] = float(best_naive_baseline_mae)

    return metrics


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Train the initial baseline ML model.

    HistGradientBoostingRegressor is a strong tabular model and a good baseline
    for lag/rolling/calendar weather features.
    """
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.0,
        early_stopping=False,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate the trained model on validation and test data."""
    valid_pred = model.predict(X_valid)
    test_pred = model.predict(X_test)

    valid_mae = mean_absolute_error(y_valid, valid_pred)
    valid_rmse = mean_squared_error(y_valid, valid_pred) ** 0.5
    valid_r2 = r2_score(y_valid, valid_pred)

    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = mean_squared_error(y_test, test_pred) ** 0.5
    test_r2 = r2_score(y_test, test_pred)

    print()
    print("Model Evaluation")
    print("----------------")
    print("Validation:")
    print(f"MAE:  {valid_mae:.4f}")
    print(f"RMSE: {valid_rmse:.4f}")
    print(f"R2:   {valid_r2:.4f}")

    print()
    print("Test:")
    print(f"MAE:  {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"R2:   {test_r2:.4f}")

    return {
        "valid_mae": float(valid_mae),
        "valid_rmse": float(valid_rmse),
        "valid_r2": float(valid_r2),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),
    }


def combine_metrics(
    model_metrics: dict,
    baseline_metrics: dict,
) -> dict:
    """Combine model metrics with naive baseline metrics."""
    metrics = {
        **model_metrics,
        **baseline_metrics,
    }

    best_baseline_mae = metrics["best_naive_baseline_mae"]
    test_mae = metrics["test_mae"]

    if best_baseline_mae > 0:
        improvement = (best_baseline_mae - test_mae) / best_baseline_mae
    else:
        improvement = 0.0

    metrics["mae_improvement_over_best_naive_baseline"] = float(improvement)

    print()
    print("Baseline comparison")
    print("-------------------")
    print(f"Best naive baseline MAE: {best_baseline_mae:.4f}")
    print(f"Model Test MAE: {test_mae:.4f}")
    print(f"MAE improvement over best naive baseline: {improvement:.2%}")

    return metrics


def build_model_bundle(model, metrics: dict) -> dict:
    """Build the serializable model bundle stored as `model.joblib`.

    The bundle contains not only the model object but also metadata required by
    the inference pipeline, especially the exact feature column order.
    """
    return {
        "model": model,
        "model_name": BASELINE_MODEL_NAME,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "metrics": metrics,
        "model_type": "HistGradientBoostingRegressor",
        "model_params": model.get_params(),
        "feature_group_name": TRAINING_FEATURE_GROUP_NAME,
        "feature_group_version": TRAINING_FEATURE_GROUP_VERSION,
        "feature_view_name": TRAINING_FEATURE_VIEW_NAME,
        "feature_view_version": TRAINING_FEATURE_VIEW_VERSION,
    }


def save_model_locally(model, metrics: dict) -> None:
    """Save the trained model bundle locally before registry upload."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_bundle = build_model_bundle(
        model=model,
        metrics=metrics,
    )

    joblib.dump(model_bundle, MODEL_LOCAL_PATH)

    print()
    print(f"Model saved locally to: {MODEL_LOCAL_PATH}")


def upload_model_to_registry(metrics: dict) -> None:
    """Upload the locally saved model bundle to Hopsworks Model Registry."""
    model_registry = get_model_registry()

    model_export_dir = MODEL_DIR / BASELINE_MODEL_NAME

    if model_export_dir.exists():
        shutil.rmtree(model_export_dir)

    model_export_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(
        MODEL_LOCAL_PATH,
        model_export_dir / "model.joblib",
    )

    input_example = {feature_name: 0.0 for feature_name in FEATURE_COLUMNS}

    metadata_path = model_export_dir / "metrics.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    model_meta = model_registry.sklearn.create_model(
        name=BASELINE_MODEL_NAME,
        metrics=metrics,
        description=(
            "Baseline HistGradientBoostingRegressor for predicting "
            "temperature 6 hours ahead using Open-Meteo weather features."
        ),
        input_example=input_example,
    )

    model_meta.save(str(model_export_dir))

    print()
    print(f"Model uploaded to Hopsworks Model Registry: {BASELINE_MODEL_NAME}")


def main(
    use_cache: bool = False,
    skip_upload: bool = False,
) -> None:
    """Run the complete baseline training workflow."""
    fs = None

    if not use_cache:
        fs = get_feature_store()

    if not use_cache:
        get_or_create_training_feature_view(fs)

    df = load_training_dataframe(
        fs=fs,
        use_cache=use_cache,
    )

    train_df, valid_df, test_df = time_based_train_valid_test_split(df)

    X_train, y_train = split_features_and_target(train_df)
    X_valid, y_valid = split_features_and_target(valid_df)
    X_test, y_test = split_features_and_target(test_df)

    baseline_metrics = evaluate_naive_baselines(
        X_test=X_test,
        y_test=y_test,
    )

    model = train_model(
        X_train=X_train,
        y_train=y_train,
    )

    model_metrics = evaluate_model(
        model=model,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
    )

    metrics = combine_metrics(
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
    )

    save_model_locally(
        model=model,
        metrics=metrics,
    )

    if skip_upload:
        print()
        print("Skipping Hopsworks Model Registry upload because --skip-upload is set.")
    else:
        upload_model_to_registry(metrics=metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train baseline temperature forecasting model."
    )

    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use local Parquet feature cache instead of reading from Hopsworks.",
    )

    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip upload to Hopsworks Model Registry.",
    )

    args = parser.parse_args()

    main(
        use_cache=args.use_cache,
        skip_upload=args.skip_upload,
    )
