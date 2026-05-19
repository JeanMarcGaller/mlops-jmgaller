"""Final training pipeline for the temperature forecasting project.

This pipeline trains the final production candidate model.

It performs the following steps:

1. Load best hyperparameters from the Optuna tuning report.
2. Load selected features from the feature-selection report.
3. Load historical training data from the Hopsworks Training Feature Group.
4. Retry Hopsworks reads if the Query Service is temporarily unstable.
5. Fall back to the local feature cache if Hopsworks cannot be read.
6. Create a chronological train/validation/test split.
7. Train the final model on train + validation data.
8. Evaluate the model on the held-out test set.
9. Compare the final model against a persistence baseline.
10. Save the model locally as a joblib bundle.
11. Upload the model artifact to the Hopsworks Model Registry.

Important design decision:
The final model uses the selected feature set from
`reports/selected_features.json`. The exact feature order is stored in the model
bundle so the inference pipeline can reproduce the same input schema.

The model uses fixed training parameters for reproducibility:
- loss="squared_error"
- early_stopping=False
- random_state=42

The remaining model parameters are loaded from the Optuna tuning report.
"""

import argparse
import json
import shutil

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    BEST_PARAMS_PATH,
    FEATURE_CACHE_PATH,
    MODEL_DIR,
    MODEL_LOCAL_PATH,
    MODEL_NAME,
    SELECTED_FEATURES_PATH,
    TARGET_COLUMN,
    TRAINING_FEATURE_GROUP_NAME,
    TRAINING_FEATURE_GROUP_VERSION,
    TRAINING_FEATURE_VIEW_NAME,
    TRAINING_FEATURE_VIEW_VERSION,
)
from hopsworks_client import get_feature_store, get_model_registry
from hopsworks_read_utils import (
    read_dataframe_from_cache,
    read_feature_group_with_retry,
)


def add_fixed_model_params(params: dict) -> dict:
    """Add fixed HistGradientBoostingRegressor parameters.

    These parameters are not tuned by Optuna. They are kept fixed for final
    training to make the model configuration reproducible and explicit.

    Args:
        params: Tuned model parameters loaded from best_params.json.

    Returns:
        Model parameters including fixed training settings.
    """
    return {
        **params,
        "loss": "squared_error",
        "early_stopping": False,
        "random_state": 42,
    }


def load_best_params() -> dict:
    """Load best Optuna hyperparameters from `best_params.json`.

    Returns:
        Best parameter dictionary for HistGradientBoostingRegressor.

    Raises:
        FileNotFoundError: If the best-params artifact does not exist.
        ValueError: If the file does not contain a `best_params` object.
    """
    if not BEST_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Best params file not found at {BEST_PARAMS_PATH}. "
            "Run `uv run python src/tuning_pipeline.py --trials 30` first."
        )

    with BEST_PARAMS_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    best_params = payload.get("best_params")

    if not best_params:
        raise ValueError(f"No 'best_params' found in {BEST_PARAMS_PATH}.")

    print()
    print("Loaded best params")
    print("------------------")
    print(best_params)

    return best_params


def load_selected_features() -> list[str]:
    """Load selected model features from `selected_features.json`.

    Returns:
        Ordered list of selected feature names.

    Raises:
        FileNotFoundError: If the selected-features artifact does not exist.
        ValueError: If the file does not contain `selected_features`.
    """
    if not SELECTED_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Selected features file not found at {SELECTED_FEATURES_PATH}. "
            "Run `uv run python src/feature_selection_pipeline.py` first."
        )

    with SELECTED_FEATURES_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    selected_features = payload.get("selected_features")

    if not selected_features:
        raise ValueError(f"No 'selected_features' found in {SELECTED_FEATURES_PATH}.")

    print()
    print("Loaded selected features")
    print("------------------------")
    print(f"Selected {len(selected_features)} features:")
    print(selected_features)

    return selected_features


def validate_training_dataframe(
    df: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    """Validate and chronologically sort the training dataframe.

    Args:
        df: Training dataframe loaded from Hopsworks or local cache.
        selected_features: Feature columns required by the final model.

    Returns:
        Validated dataframe sorted by event_time.

    Raises:
        ValueError: If target, selected features or event_time are missing.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    missing_features = [
        feature for feature in selected_features if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(f"Missing selected feature columns: {missing_features}")

    if "event_time" not in df.columns:
        raise ValueError("Missing event_time column for time-based split.")

    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time").reset_index(drop=True)

    return df


def load_training_dataframe(
    fs,
    selected_features: list[str],
    use_cache: bool = False,
) -> pd.DataFrame:
    """Load training data from Hopsworks or local cache."""
    if use_cache:
        print("Loading training data from local cache because --use-cache is set.")

        df = read_dataframe_from_cache(FEATURE_CACHE_PATH)

        return validate_training_dataframe(
            df=df,
            selected_features=selected_features,
        )

    print("Loading training data from Hopsworks Feature Group with robust read...")

    feature_group = fs.get_feature_group(
        name=TRAINING_FEATURE_GROUP_NAME,
        version=TRAINING_FEATURE_GROUP_VERSION,
    )

    return read_feature_group_with_retry(
        feature_group=feature_group,
        cache_path=FEATURE_CACHE_PATH,
        validate_fn=lambda df: validate_training_dataframe(
            df=df,
            selected_features=selected_features,
        ),
        context="final training",
    )


def time_based_train_valid_test_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    valid_fraction: float = 0.15,
):
    """Split data chronologically into train, validation and test sets.

    Final training fits on train + validation and evaluates once on test.

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


def split_features_and_target(
    df: pd.DataFrame,
    feature_columns: list[str],
):
    """Split dataframe into model input features and target."""
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def train_final_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    selected_features: list[str],
    best_params: dict,
):
    """Train final model on train + validation data.

    Args:
        train_df: Training split.
        valid_df: Validation split.
        selected_features: Ordered feature list for final model input.
        best_params: Best Optuna hyperparameters.

    Returns:
        Fitted HistGradientBoostingRegressor.
    """
    train_valid_df = pd.concat(
        [train_df, valid_df],
        axis=0,
        ignore_index=True,
    )

    X_train_valid, y_train_valid = split_features_and_target(
        train_valid_df,
        selected_features,
    )

    params = add_fixed_model_params(best_params)

    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train_valid, y_train_valid)

    return model


def evaluate_naive_baselines(
    test_df: pd.DataFrame,
) -> dict:
    """Evaluate naive forecasting baselines on the test set.

    The persistence baseline predicts that the future temperature equals the
    current temperature.

    The seasonal naive baseline predicts that the future temperature equals the
    temperature at the same feature time one day before.
    """
    y_test = test_df[TARGET_COLUMN]

    baseline_predictions = {
        "persistence": test_df["temperature_2m"],
        "seasonal_naive_24h": test_df["temperature_lag_24h"],
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


def evaluate_final_model(
    model,
    test_df: pd.DataFrame,
    selected_features: list[str],
) -> dict:
    """Evaluate final model on the held-out test set."""
    X_test, y_test = split_features_and_target(
        test_df,
        selected_features,
    )

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print()
    print("Final Model Evaluation")
    print("----------------------")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2:   {r2:.4f}")

    return {
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "test_r2": float(r2),
    }


def combine_metrics(
    final_metrics: dict,
    baseline_metrics: dict,
    selected_features: list[str],
) -> dict:
    """Combine final model, naive baseline and metadata metrics."""
    metrics = {
        **final_metrics,
        **baseline_metrics,
        "n_selected_features": len(selected_features),
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
    print(f"Final Model Test MAE: {test_mae:.4f}")
    print(f"MAE improvement over best naive baseline: {improvement:.2%}")

    return metrics


def build_model_bundle(
    model,
    selected_features: list[str],
    best_params: dict,
    metrics: dict,
) -> dict:
    """Build the serializable model bundle stored as model.joblib.

    The inference pipeline depends on this metadata, especially the exact
    selected feature order.
    """
    return {
        "model": model,
        "model_name": MODEL_NAME,
        "model_stage": "final",
        "feature_columns": selected_features,
        "target_column": TARGET_COLUMN,
        "metrics": metrics,
        "model_type": "HistGradientBoostingRegressor",
        "model_params": model.get_params(),
        "best_params": best_params,
        "feature_selection": "permutation_importance",
        "feature_group_name": TRAINING_FEATURE_GROUP_NAME,
        "feature_group_version": TRAINING_FEATURE_GROUP_VERSION,
        "feature_view_name": TRAINING_FEATURE_VIEW_NAME,
        "feature_view_version": TRAINING_FEATURE_VIEW_VERSION,
    }


def save_model_locally(
    model,
    selected_features: list[str],
    best_params: dict,
    metrics: dict,
) -> None:
    """Save the final model bundle locally before registry upload."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_bundle = build_model_bundle(
        model=model,
        selected_features=selected_features,
        best_params=best_params,
        metrics=metrics,
    )

    joblib.dump(model_bundle, MODEL_LOCAL_PATH)

    print()
    print(f"Model saved locally to: {MODEL_LOCAL_PATH}")


def upload_model_to_registry(
    selected_features: list[str],
    metrics: dict,
) -> None:
    """Upload the final model artifact to Hopsworks Model Registry."""
    model_registry = get_model_registry()

    model_export_dir = MODEL_DIR / MODEL_NAME

    if model_export_dir.exists():
        shutil.rmtree(model_export_dir)

    model_export_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(
        MODEL_LOCAL_PATH,
        model_export_dir / "model.joblib",
    )

    input_example = {feature_name: 0.0 for feature_name in selected_features}

    metadata_path = model_export_dir / "metrics.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    selected_features_path = model_export_dir / "selected_features.json"
    with selected_features_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "selected_features": selected_features,
                "target_column": TARGET_COLUMN,
            },
            file,
            indent=2,
        )

    model_meta = model_registry.sklearn.create_model(
        name=MODEL_NAME,
        metrics=metrics,
        description=(
            "Final HistGradientBoostingRegressor for predicting temperature "
            "6 hours ahead using selected Open-Meteo weather features."
        ),
        input_example=input_example,
    )

    model_meta.save(str(model_export_dir))

    print()
    print(f"Model uploaded to Hopsworks Model Registry: {MODEL_NAME}")


def main(
    use_cache: bool = False,
    skip_upload: bool = False,
) -> None:
    """Run the complete final training workflow."""
    best_params = load_best_params()
    selected_features = load_selected_features()

    fs = None

    if not use_cache:
        fs = get_feature_store()

    df = load_training_dataframe(
        fs=fs,
        selected_features=selected_features,
        use_cache=use_cache,
    )

    train_df, valid_df, test_df = time_based_train_valid_test_split(df)

    model = train_final_model(
        train_df=train_df,
        valid_df=valid_df,
        selected_features=selected_features,
        best_params=best_params,
    )

    baseline_metrics = evaluate_naive_baselines(test_df)

    final_metrics = evaluate_final_model(
        model=model,
        test_df=test_df,
        selected_features=selected_features,
    )

    metrics = combine_metrics(
        final_metrics=final_metrics,
        baseline_metrics=baseline_metrics,
        selected_features=selected_features,
    )

    save_model_locally(
        model=model,
        selected_features=selected_features,
        best_params=best_params,
        metrics=metrics,
    )

    if skip_upload:
        print()
        print("Skipping Hopsworks Model Registry upload because --skip-upload is set.")
    else:
        upload_model_to_registry(
            selected_features=selected_features,
            metrics=metrics,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train final temperature forecasting model."
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
