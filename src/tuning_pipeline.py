"""Hyperparameter tuning pipeline for the temperature forecasting project.

This pipeline tunes a HistGradientBoostingRegressor with Optuna.

It performs the following steps:

1. Load historical training data from the Hopsworks Training Feature Group.
2. Retry Hopsworks reads if the Query Service is temporarily unstable.
3. Fall back to the local feature cache if Hopsworks cannot be read.
4. Create a chronological train/validation/test split.
5. Optimize hyperparameters on the validation set using Optuna.
6. Train the best parameter set on train + validation data.
7. Evaluate the tuned model once on the held-out test set.
8. Save tuning artifacts locally:
   - reports/best_params.json
   - reports/tuning_results.csv

Important design decision:
The test set is not used during hyperparameter search. It is only used once
after tuning to estimate generalization performance.

The model uses fixed training parameters for reproducibility:
- loss="squared_error"
- early_stopping=False
- random_state=42

Only the remaining hyperparameters are optimized by Optuna.
"""

import argparse
import json

import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    BEST_PARAMS_PATH,
    FEATURE_CACHE_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
    TRAINING_FEATURE_GROUP_NAME,
    TRAINING_FEATURE_GROUP_VERSION,
    TUNING_RESULTS_PATH,
)
from features import FEATURE_COLUMNS
from hopsworks_client import get_feature_store
from hopsworks_read_utils import (
    read_dataframe_from_cache,
    read_feature_group_with_retry,
)


def add_fixed_model_params(params: dict) -> dict:
    """Add fixed HistGradientBoostingRegressor parameters.

    These parameters are not tuned by Optuna. They are kept fixed across all
    tuning trials and the final tuned model for reproducibility and consistency.

    Args:
        params: Tunable model parameters.

    Returns:
        Model parameters including fixed training settings.
    """
    return {
        **params,
        "loss": "squared_error",
        "early_stopping": False,
        "random_state": 42,
    }


def load_training_dataframe(
    fs,
    use_cache: bool = False,
) -> pd.DataFrame:
    """Load training data from Hopsworks or local cache.

    Args:
        fs: Hopsworks Feature Store object.
        use_cache: If true, skip Hopsworks and read from local Parquet cache.

    Returns:
        Validated and chronologically sorted training dataframe.
    """
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
        context="offline hyperparameter tuning",
    )


def validate_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and chronologically sort the training dataframe.

    Args:
        df: Training dataframe loaded from Hopsworks or local cache.

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


def time_based_train_valid_test_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    valid_fraction: float = 0.15,
):
    """Split data chronologically into train, validation and test sets.

    The validation set is used by Optuna to choose hyperparameters.
    The test set is held back for one final evaluation after tuning.

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
    """Split dataframe into model features and target."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def create_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
):
    """Create the Optuna objective function.

    The objective trains one model per trial and returns validation MAE.
    Optuna minimizes this value.

    Args:
        X_train: Training feature matrix.
        y_train: Training target.
        X_valid: Validation feature matrix.
        y_valid: Validation target.

    Returns:
        Callable Optuna objective.
    """

    def objective(trial: optuna.Trial) -> float:
        params = add_fixed_model_params(
            {
                "max_iter": trial.suggest_int("max_iter", 100, 600),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.2,
                    log=True,
                ),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    10,
                    80,
                ),
                "l2_regularization": trial.suggest_float(
                    "l2_regularization",
                    1e-5,
                    10.0,
                    log=True,
                ),
                "max_bins": trial.suggest_int("max_bins", 64, 255),
            }
        )

        model = HistGradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        valid_pred = model.predict(X_valid)
        valid_mae = mean_absolute_error(y_valid, valid_pred)

        return valid_mae

    return objective


def run_optuna_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    n_trials: int,
) -> optuna.Study:
    """Run Optuna hyperparameter tuning.

    Args:
        X_train: Training feature matrix.
        y_train: Training target.
        X_valid: Validation feature matrix.
        y_valid: Validation target.
        n_trials: Number of Optuna trials.

    Returns:
        Completed Optuna study.
    """
    objective = create_objective(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
    )

    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        direction="minimize",
        study_name="temperature_forecast_hgb_tuning",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print()
    print("Optuna tuning finished")
    print("----------------------")
    print(f"Best validation MAE: {study.best_value:.4f}")
    print("Best params:")
    print(study.best_params)

    return study


def train_best_model_and_evaluate(
    study: optuna.Study,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Train best tuned model on train + validation and evaluate on test.

    Args:
        study: Completed Optuna study.
        train_df: Training split.
        valid_df: Validation split.
        test_df: Held-out test split.

    Returns:
        Dictionary with validation and test metrics.
    """
    train_valid_df = pd.concat(
        [train_df, valid_df],
        axis=0,
        ignore_index=True,
    )

    X_train_valid, y_train_valid = split_features_and_target(train_valid_df)
    X_test, y_test = split_features_and_target(test_df)

    best_params = add_fixed_model_params(study.best_params)

    model = HistGradientBoostingRegressor(**best_params)
    model.fit(X_train_valid, y_train_valid)

    test_pred = model.predict(X_test)

    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = mean_squared_error(y_test, test_pred) ** 0.5
    test_r2 = r2_score(y_test, test_pred)

    print()
    print("Best tuned model evaluation")
    print("---------------------------")
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R2:   {test_r2:.4f}")

    return {
        "best_valid_mae": float(study.best_value),
        "tuned_test_mae": float(test_mae),
        "tuned_test_rmse": float(test_rmse),
        "tuned_test_r2": float(test_r2),
    }


def build_best_params_payload(
    study: optuna.Study,
    evaluation_metrics: dict,
) -> dict:
    """Build JSON-serializable tuning summary payload."""
    return {
        "model_type": "HistGradientBoostingRegressor",
        "best_params": study.best_params,
        "final_model_params": add_fixed_model_params(study.best_params),
        "best_valid_mae": float(study.best_value),
        "evaluation_metrics": evaluation_metrics,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
    }


def save_tuning_artifacts(
    study: optuna.Study,
    evaluation_metrics: dict,
) -> None:
    """Save best parameters and all Optuna trials locally.

    These reports are experiment artifacts and should not be committed to Git.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    best_params_payload = build_best_params_payload(
        study=study,
        evaluation_metrics=evaluation_metrics,
    )

    with BEST_PARAMS_PATH.open("w", encoding="utf-8") as file:
        json.dump(best_params_payload, file, indent=2)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(TUNING_RESULTS_PATH, index=False)

    print()
    print("Saved tuning artifacts")
    print("----------------------")
    print(f"Best params: {BEST_PARAMS_PATH}")
    print(f"Tuning results: {TUNING_RESULTS_PATH}")


def main(
    n_trials: int,
    use_cache: bool = False,
) -> None:
    """Run the full hyperparameter tuning workflow."""
    if n_trials <= 0:
        raise ValueError("n_trials must be greater than 0.")

    fs = None

    if not use_cache:
        fs = get_feature_store()

    df = load_training_dataframe(
        fs=fs,
        use_cache=use_cache,
    )

    train_df, valid_df, test_df = time_based_train_valid_test_split(df)

    X_train, y_train = split_features_and_target(train_df)
    X_valid, y_valid = split_features_and_target(valid_df)

    study = run_optuna_tuning(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        n_trials=n_trials,
    )

    evaluation_metrics = train_best_model_and_evaluate(
        study=study,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )

    save_tuning_artifacts(
        study=study,
        evaluation_metrics=evaluation_metrics,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune HistGradientBoostingRegressor hyperparameters with Optuna."
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of Optuna trials.",
    )

    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use local Parquet feature cache instead of reading from Hopsworks.",
    )

    args = parser.parse_args()

    main(
        n_trials=args.trials,
        use_cache=args.use_cache,
    )
