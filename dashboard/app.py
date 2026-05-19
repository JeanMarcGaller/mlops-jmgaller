"""Streamlit dashboard for the MLOps temperature forecasting project.

The dashboard reads local report artifacts created by the pipelines:

- reports/latest_prediction.json
- reports/prediction_log.csv
- reports/monitoring_results.csv
- reports/backtesting_results.csv
- reports/permutation_importance.csv
- reports/selected_features.json
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

LATEST_PREDICTION_PATH = REPORTS_DIR / "latest_prediction.json"
PREDICTION_LOG_PATH = REPORTS_DIR / "prediction_log.csv"
MONITORING_RESULTS_PATH = REPORTS_DIR / "monitoring_results.csv"
BACKTESTING_RESULTS_PATH = REPORTS_DIR / "backtesting_results.csv"
PERMUTATION_IMPORTANCE_PATH = REPORTS_DIR / "permutation_importance.csv"
SELECTED_FEATURES_PATH = REPORTS_DIR / "selected_features.json"
MODEL_COMPARISON_RESULTS_PATH = REPORTS_DIR / "model_comparison_results.csv"


st.set_page_config(
    page_title="Temperature Forecasting Dashboard",
    page_icon="🌡️",
    layout="wide",
)


def read_json(path: Path) -> dict | None:
    """Read a JSON file if it exists."""
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file if it exists."""
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def show_missing_artifact_message(path: Path) -> None:
    """Show a consistent missing artifact message."""
    st.info(f"Artifact not found yet: `{path.relative_to(PROJECT_ROOT)}`")


def render_latest_prediction() -> None:
    """Render latest prediction card."""
    st.header("Latest Prediction")

    latest_prediction = read_json(LATEST_PREDICTION_PATH)

    if latest_prediction is None:
        show_missing_artifact_message(LATEST_PREDICTION_PATH)
        return

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label="Predicted temperature",
        value=f"{latest_prediction['predicted_temperature_celsius']:.2f} °C",
    )
    col2.metric(
        label="Location",
        value=latest_prediction.get("location_id", "unknown"),
    )
    col3.metric(
        label="Model version",
        value=latest_prediction.get("model_version", "unknown"),
    )
    col4.metric(
        label="Model test MAE",
        value=latest_prediction.get("model_test_mae", "n/a"),
    )

    st.write("Prediction metadata")
    st.json(latest_prediction)


def render_prediction_log() -> None:
    """Render prediction log section."""
    st.header("Prediction Log")

    prediction_log_df = read_csv(PREDICTION_LOG_PATH)

    if prediction_log_df.empty:
        show_missing_artifact_message(PREDICTION_LOG_PATH)
        return

    if "predicted_event_time" in prediction_log_df.columns:
        prediction_log_df["predicted_event_time"] = pd.to_datetime(
            prediction_log_df["predicted_event_time"]
        )
        prediction_log_df = prediction_log_df.sort_values("predicted_event_time")

    st.dataframe(prediction_log_df, use_container_width=True)

    if {
        "predicted_event_time",
        "predicted_temperature_celsius",
    }.issubset(prediction_log_df.columns):
        chart_df = prediction_log_df.set_index("predicted_event_time")[
            ["predicted_temperature_celsius"]
        ]
        st.line_chart(chart_df)


def render_monitoring() -> None:
    """Render monitoring results section."""
    st.header("Monitoring")

    monitoring_df = read_csv(MONITORING_RESULTS_PATH)

    if monitoring_df.empty:
        show_missing_artifact_message(MONITORING_RESULTS_PATH)
        st.caption(
            "Monitoring results are available only after the predicted timestamp "
            "is available in the historical weather API."
        )
        return

    if "predicted_event_time" in monitoring_df.columns:
        monitoring_df["predicted_event_time"] = pd.to_datetime(
            monitoring_df["predicted_event_time"]
        )
        monitoring_df = monitoring_df.sort_values("predicted_event_time")

    latest_row = monitoring_df.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matched predictions", len(monitoring_df))
    col2.metric(
        "Mean absolute error",
        f"{monitoring_df['absolute_error_celsius'].mean():.2f} °C",
    )
    col3.metric(
        "Latest absolute error",
        f"{latest_row['absolute_error_celsius']:.2f} °C",
    )

    if "rolling_mae_7_predictions" in monitoring_df.columns:
        col4.metric(
            "Rolling MAE",
            f"{latest_row['rolling_mae_7_predictions']:.2f} °C",
        )

    st.dataframe(monitoring_df, use_container_width=True)

    chart_columns = [
        column
        for column in [
            "predicted_temperature_celsius",
            "actual_temperature_celsius",
        ]
        if column in monitoring_df.columns
    ]

    if chart_columns and "predicted_event_time" in monitoring_df.columns:
        st.subheader("Predicted vs actual temperature")
        chart_df = monitoring_df.set_index("predicted_event_time")[chart_columns]
        st.line_chart(chart_df)

    if {
        "predicted_event_time",
        "absolute_error_celsius",
    }.issubset(monitoring_df.columns):
        st.subheader("Absolute error")
        error_df = monitoring_df.set_index("predicted_event_time")[
            ["absolute_error_celsius"]
        ]
        st.line_chart(error_df)


def render_backtesting() -> None:
    """Render backtesting results section."""
    st.header("Backtesting")

    backtesting_df = read_csv(BACKTESTING_RESULTS_PATH)

    if backtesting_df.empty:
        show_missing_artifact_message(BACKTESTING_RESULTS_PATH)
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Backtest splits", len(backtesting_df))
    col2.metric("Mean model MAE", f"{backtesting_df['model_mae'].mean():.2f}")
    col3.metric(
        "Mean best naive MAE",
        f"{backtesting_df['best_naive_baseline_mae'].mean():.2f}",
    )

    st.dataframe(backtesting_df, use_container_width=True)

    if {
        "split_id",
        "model_mae",
        "best_naive_baseline_mae",
    }.issubset(backtesting_df.columns):
        chart_df = backtesting_df.set_index("split_id")[
            [
                "model_mae",
                "best_naive_baseline_mae",
            ]
        ]
        st.line_chart(chart_df)


def render_feature_selection() -> None:
    """Render feature selection section."""
    st.header("Feature Selection")

    selected_features = read_json(SELECTED_FEATURES_PATH)
    importance_df = read_csv(PERMUTATION_IMPORTANCE_PATH)

    if selected_features is None:
        show_missing_artifact_message(SELECTED_FEATURES_PATH)
    else:
        col1, col2 = st.columns(2)
        col1.metric(
            "Selected features",
            selected_features.get("n_selected_features", "n/a"),
        )
        col2.metric(
            "All features",
            selected_features.get("n_all_features", "n/a"),
        )

        st.write("Selected feature list")
        st.write(selected_features.get("selected_features", []))

    if importance_df.empty:
        show_missing_artifact_message(PERMUTATION_IMPORTANCE_PATH)
        return

    st.subheader("Permutation importance")
    st.dataframe(importance_df, use_container_width=True)

    if {"feature", "importance_mean"}.issubset(importance_df.columns):
        top_importance_df = (
            importance_df.sort_values("importance_mean", ascending=False)
            .head(15)
            .set_index("feature")
        )

        st.bar_chart(top_importance_df[["importance_mean"]])


def render_model_comparison() -> None:
    """Render model comparison results section."""
    st.header("Model Comparison")

    comparison_df = read_csv(MODEL_COMPARISON_RESULTS_PATH)

    if comparison_df.empty:
        show_missing_artifact_message(MODEL_COMPARISON_RESULTS_PATH)
        return

    ok_df = comparison_df.loc[comparison_df["status"] == "ok"].copy()

    if ok_df.empty:
        st.warning("No successful model comparison results available.")
        st.dataframe(comparison_df, use_container_width=True)
        return

    best_row = ok_df.sort_values("test_mae").iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Best model", best_row["model_name"])
    col2.metric("Best test MAE", f"{best_row['test_mae']:.4f}")
    col3.metric("Best test R2", f"{best_row['test_r2']:.4f}")

    st.dataframe(comparison_df, use_container_width=True)

    chart_df = ok_df.sort_values("test_mae").set_index("model_name")[
        [
            "valid_mae",
            "test_mae",
        ]
    ]

    st.subheader("Validation vs Test MAE")
    st.bar_chart(chart_df)

    if {"model_name", "fit_seconds"}.issubset(ok_df.columns):
        fit_time_df = ok_df.set_index("model_name")[["fit_seconds"]]
        st.subheader("Fit time")
        st.bar_chart(fit_time_df)


def main() -> None:
    """Run Streamlit dashboard."""
    st.title("🌡️ MLOps Temperature Forecasting Dashboard")
    st.caption("Local dashboard based on pipeline report artifacts.")

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select page",
            [
                "Overview",
                "Prediction Log",
                "Monitoring",
                "Backtesting",
                "Model Comparison",
                "Feature Selection",
            ],
        )

        st.divider()
        st.write("Reports directory")
        st.code(str(REPORTS_DIR))

    if page == "Overview":
        render_latest_prediction()
        st.divider()
        render_monitoring()
    elif page == "Prediction Log":
        render_prediction_log()
    elif page == "Monitoring":
        render_monitoring()
    elif page == "Backtesting":
        render_backtesting()
    elif page == "Model Comparison":
        render_model_comparison()
    elif page == "Feature Selection":
        render_feature_selection()


if __name__ == "__main__":
    main()
