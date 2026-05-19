"""Feature selection pipeline for the temperature forecasting project.

This pipeline selects a smaller set of relevant model features after Optuna
hyperparameter tuning.

It performs the following steps:

1. Load historical training data from the Hopsworks Training Feature Group.
2. Retry Hopsworks reads if the Query Service is temporarily unstable.
3. Fall back to the local feature cache if Hopsworks cannot be read.
4. Load the best hyperparameters from `reports/best_params.json`.
5. Compute correlation reports for feature diagnostics.
6. Add random probe features for feature-importance calibration.
7. Train a HistGradientBoostingRegressor on the training split.
8. Compute permutation importance on the validation split.
9. Select real features that outperform random probe features.
10. Save feature-selection artifacts locally.

Important design decision:
Feature selection is performed on the validation set, not the test set. The test
set should remain reserved for final unbiased model evaluation.

Random probe features are used only inside this pipeline. They are not written
to Hopsworks, not used for inference and never included in selected_features.json.
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    BEST_PARAMS_PATH,
    FEATURE_CACHE_PATH,
    FEATURE_CORRELATION_MATRIX_PATH,
    HIGH_CORRELATION_PAIRS_PATH,
    PERMUTATION_IMPORTANCE_PATH,
    REPORTS_DIR,
    SELECTED_FEATURES_PATH,
    TARGET_COLUMN,
    TARGET_CORRELATION_PATH,
    TRAINING_FEATURE_GROUP_NAME,
    TRAINING_FEATURE_GROUP_VERSION,
)
from features import FEATURE_COLUMNS
from hopsworks_client import get_feature_store
from hopsworks_read_utils import (
    read_dataframe_from_cache,
    read_feature_group_with_retry,
)

# Fallback rule:
# If fewer than MIN_SELECTED_FEATURES outperform the random probes, keep the top
# real features by permutation importance.
MIN_SELECTED_FEATURES = 8

RANDOM_FEATURE_COLUMNS = [
    "random_normal_probe",
    "random_uniform_probe",
    "random_permuted_temperature_probe",
]

CORRELATION_THRESHOLD = 0.90


def add_fixed_model_params(params: dict) -> dict:
    """Add fixed HistGradientBoostingRegressor parameters.

    These parameters are kept consistent with tuning and final training.
    """
    return {
        **params,
        "loss": "squared_error",
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
        raise ValueError("Missing event_time column for time-based split.")

    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time").reset_index(drop=True)

    return df


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
        context="feature selection",
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
    """Split dataframe into selected model features and target."""
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def add_random_probe_features(
    df: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Add random probe features for feature-importance calibration.

    Random probe features intentionally contain no useful signal. They are used
    only during feature selection to estimate how important pure noise can look.
    """
    df = df.copy()
    rng = np.random.default_rng(random_state)

    df["random_normal_probe"] = rng.normal(size=len(df))
    df["random_uniform_probe"] = rng.uniform(size=len(df))
    df["random_permuted_temperature_probe"] = rng.permutation(
        df["temperature_2m"].to_numpy()
    )

    return df


def compute_correlation_reports(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float = CORRELATION_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute feature correlation reports for feature-selection diagnostics."""
    correlation_matrix = train_df[feature_columns].corr(numeric_only=True)

    high_correlation_pairs = []
    abs_correlation_matrix = correlation_matrix.abs()

    for feature_index, feature_a in enumerate(feature_columns):
        for feature_b in feature_columns[feature_index + 1 :]:
            abs_correlation = abs_correlation_matrix.loc[feature_a, feature_b]

            if abs_correlation >= threshold:
                correlation = correlation_matrix.loc[feature_a, feature_b]
                high_correlation_pairs.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "correlation": float(correlation),
                        "abs_correlation": float(abs_correlation),
                    }
                )

    high_correlation_df = pd.DataFrame(
        high_correlation_pairs,
        columns=[
            "feature_a",
            "feature_b",
            "correlation",
            "abs_correlation",
        ],
    )

    if not high_correlation_df.empty:
        high_correlation_df = high_correlation_df.sort_values(
            "abs_correlation",
            ascending=False,
        ).reset_index(drop=True)

    target_correlation_df = (
        train_df[feature_columns + [TARGET_COLUMN]]
        .corr(numeric_only=True)[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .reset_index()
        .rename(columns={"index": "feature", TARGET_COLUMN: "target_correlation"})
    )

    target_correlation_df["abs_target_correlation"] = target_correlation_df[
        "target_correlation"
    ].abs()

    target_correlation_df = target_correlation_df.sort_values(
        "abs_target_correlation",
        ascending=False,
    ).reset_index(drop=True)

    return correlation_matrix, high_correlation_df, target_correlation_df


def save_correlation_reports(
    correlation_matrix: pd.DataFrame,
    high_correlation_df: pd.DataFrame,
    target_correlation_df: pd.DataFrame,
) -> None:
    """Save correlation diagnostics to local report files."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    correlation_matrix.to_csv(FEATURE_CORRELATION_MATRIX_PATH)
    high_correlation_df.to_csv(HIGH_CORRELATION_PAIRS_PATH, index=False)
    target_correlation_df.to_csv(TARGET_CORRELATION_PATH, index=False)

    print()
    print("Saved correlation reports")
    print("-------------------------")
    print(f"Feature correlation matrix: {FEATURE_CORRELATION_MATRIX_PATH}")
    print(f"Highly correlated feature pairs: {HIGH_CORRELATION_PAIRS_PATH}")
    print(f"Target correlation: {TARGET_CORRELATION_PATH}")


def load_best_params() -> dict:
    """Load best Optuna hyperparameters from the tuning report."""
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


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
):
    """Train a model using the best Optuna hyperparameters."""
    params = add_fixed_model_params(best_params)

    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> dict:
    """Evaluate the tuned model on the validation set before feature selection."""
    y_pred = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, y_pred)
    rmse = mean_squared_error(y_valid, y_pred) ** 0.5
    r2 = r2_score(y_valid, y_pred)

    print()
    print("Validation evaluation before feature selection")
    print("----------------------------------------------")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    return {
        "valid_mae_before_selection": float(mae),
        "valid_rmse_before_selection": float(rmse),
        "valid_r2_before_selection": float(r2),
    }


def compute_permutation_importance(
    model,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> pd.DataFrame:
    """Compute permutation importance on validation data."""
    print()
    print("Computing permutation importance...")
    print("Scoring: neg_mean_absolute_error")

    result = permutation_importance(
        estimator=model,
        X=X_valid,
        y=y_valid,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_valid.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    importance_df = importance_df.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    print()
    print("Permutation importance")
    print("----------------------")
    print(importance_df)

    return importance_df


def get_random_probe_threshold(importance_df: pd.DataFrame) -> float:
    """Return the strongest random probe importance as noise threshold."""
    random_probe_mask = importance_df["feature"].isin(RANDOM_FEATURE_COLUMNS)

    if not random_probe_mask.any():
        raise ValueError(
            "Permutation importance does not contain random probe features."
        )

    random_probe_threshold = importance_df.loc[
        random_probe_mask,
        "importance_mean",
    ].max()

    return float(random_probe_threshold)


def select_relevant_features(
    importance_df: pd.DataFrame,
) -> list[str]:
    """Select real features that outperform random probe features."""
    random_probe_threshold = get_random_probe_threshold(importance_df)
    real_features_df = importance_df.loc[
        ~importance_df["feature"].isin(RANDOM_FEATURE_COLUMNS)
    ].copy()

    selected_features = real_features_df.loc[
        real_features_df["importance_mean"] > random_probe_threshold,
        "feature",
    ].tolist()

    if len(selected_features) < MIN_SELECTED_FEATURES:
        selected_features = real_features_df.head(MIN_SELECTED_FEATURES)[
            "feature"
        ].tolist()

    print()
    print("Selected features")
    print("-----------------")
    print(f"Random probe threshold: {random_probe_threshold:.6f}")
    print(f"Selected {len(selected_features)} of {len(FEATURE_COLUMNS)} real features:")
    print(selected_features)

    return selected_features


def build_selected_features_payload(
    importance_df: pd.DataFrame,
    selected_features: list[str],
    metrics: dict,
    best_params: dict,
) -> dict:
    """Build the JSON-serializable selected-features report payload."""
    random_probe_importance_df = importance_df.loc[
        importance_df["feature"].isin(RANDOM_FEATURE_COLUMNS)
    ].copy()

    random_probe_threshold = get_random_probe_threshold(importance_df)

    return {
        "model_type": "HistGradientBoostingRegressor",
        "target_column": TARGET_COLUMN,
        "all_features": FEATURE_COLUMNS,
        "selected_features": selected_features,
        "n_all_features": len(FEATURE_COLUMNS),
        "n_selected_features": len(selected_features),
        "selection_method": "permutation_importance_with_random_probes",
        "random_probe_features": RANDOM_FEATURE_COLUMNS,
        "random_probe_threshold": random_probe_threshold,
        "random_probe_importance": random_probe_importance_df.to_dict(orient="records"),
        "min_selected_features": MIN_SELECTED_FEATURES,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "best_params": best_params,
        "model_params": add_fixed_model_params(best_params),
        "metrics": metrics,
    }


def save_feature_selection_artifacts(
    importance_df: pd.DataFrame,
    selected_features: list[str],
    metrics: dict,
    best_params: dict,
) -> None:
    """Save permutation importance and selected-features report locally."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    importance_df.to_csv(PERMUTATION_IMPORTANCE_PATH, index=False)

    payload = build_selected_features_payload(
        importance_df=importance_df,
        selected_features=selected_features,
        metrics=metrics,
        best_params=best_params,
    )

    with SELECTED_FEATURES_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print()
    print("Saved feature selection artifacts")
    print("---------------------------------")
    print(f"Permutation importance: {PERMUTATION_IMPORTANCE_PATH}")
    print(f"Selected features: {SELECTED_FEATURES_PATH}")


def main(use_cache: bool = False) -> None:
    """Run the complete feature-selection workflow."""
    fs = None

    if not use_cache:
        fs = get_feature_store()

    df = load_training_dataframe(
        fs=fs,
        use_cache=use_cache,
    )

    train_df, valid_df, _test_df = time_based_train_valid_test_split(df)

    best_params = load_best_params()

    correlation_matrix, high_correlation_df, target_correlation_df = (
        compute_correlation_reports(
            train_df=train_df,
            feature_columns=FEATURE_COLUMNS,
        )
    )

    save_correlation_reports(
        correlation_matrix=correlation_matrix,
        high_correlation_df=high_correlation_df,
        target_correlation_df=target_correlation_df,
    )

    train_df = add_random_probe_features(train_df, random_state=42)
    valid_df = add_random_probe_features(valid_df, random_state=43)

    feature_columns_with_probes = FEATURE_COLUMNS + RANDOM_FEATURE_COLUMNS

    X_train, y_train = split_features_and_target(
        train_df,
        feature_columns_with_probes,
    )
    X_valid, y_valid = split_features_and_target(
        valid_df,
        feature_columns_with_probes,
    )

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        best_params=best_params,
    )

    metrics = evaluate_model(
        model=model,
        X_valid=X_valid,
        y_valid=y_valid,
    )

    importance_df = compute_permutation_importance(
        model=model,
        X_valid=X_valid,
        y_valid=y_valid,
    )

    selected_features = select_relevant_features(importance_df)

    save_feature_selection_artifacts(
        importance_df=importance_df,
        selected_features=selected_features,
        metrics=metrics,
        best_params=best_params,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select relevant model features with permutation importance."
    )

    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use local Parquet feature cache instead of reading from Hopsworks.",
    )

    args = parser.parse_args()

    main(use_cache=args.use_cache)
