"""Streamlit dashboard for blast-vibration model review."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from predict import DEFAULT_MODEL_PATH, prepare_prediction_data

try:
    import streamlit as st
except Exception:  # pragma: no cover - optional UI dependency
    st = None


DEFAULT_DATA_PATH = Path("data/ground_vibration_dataset.csv")
FEATURE_IMPORTANCE_PATH = Path("artifacts/feature_importance.csv")


def run_dashboard() -> None:
    if st is None:
        raise RuntimeError("Streamlit is not installed. Run `pip install streamlit`.")

    st.set_page_config(page_title="MineralForge", layout="wide")
    st.title("MineralForge Blast Vibration Dashboard")

    uploaded = st.sidebar.file_uploader("Upload blast vibration CSV", type=["csv"])
    frame = pd.read_csv(uploaded) if uploaded else pd.read_csv(DEFAULT_DATA_PATH)

    model = joblib.load(DEFAULT_MODEL_PATH)
    x = prepare_prediction_data_from_frame(frame)
    predictions = model.predict(x)
    result = frame.copy()
    result["Predicted_Vibration_Level"] = predictions

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        for label, probability in zip(model.classes_, probabilities.T):
            result[f"Probability_{label}"] = probability

    high_count = int((result["Predicted_Vibration_Level"] == "High").sum())
    medium_count = int((result["Predicted_Vibration_Level"] == "Medium").sum())
    ppv_average = float(result["PPV(mm/s)"].mean()) if "PPV(mm/s)" in result.columns else 0.0

    left, middle, right = st.columns(3)
    left.metric("Predicted high events", high_count)
    middle.metric("Predicted medium events", medium_count)
    right.metric("Average PPV", f"{ppv_average:.2f} mm/s")

    if "Timestamp" in result.columns and "PPV(mm/s)" in result.columns:
        st.subheader("PPV timeline")
        timeline = result.copy()
        timeline["Timestamp"] = pd.to_datetime(timeline["Timestamp"], errors="coerce")
        st.line_chart(timeline.set_index("Timestamp")["PPV(mm/s)"])

    if {"PPV(mm/s)", "Frequency(Hz)"}.issubset(result.columns):
        st.subheader("PPV vs frequency")
        st.scatter_chart(result, x="Frequency(Hz)", y="PPV(mm/s)", color="Predicted_Vibration_Level")

    if FEATURE_IMPORTANCE_PATH.exists():
        st.subheader("Feature importance")
        importance = pd.read_csv(FEATURE_IMPORTANCE_PATH).head(15)
        st.bar_chart(importance, x="feature", y="importance")

    st.subheader("Predictions")
    st.dataframe(result, use_container_width=True)


def prepare_prediction_data_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    temp_path = Path("artifacts/.dashboard_input.csv")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(temp_path, index=False)
    try:
        return prepare_prediction_data(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    run_dashboard()
