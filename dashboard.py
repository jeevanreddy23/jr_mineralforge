"""Streamlit dashboard for MineralForge field review."""

from __future__ import annotations

import pandas as pd

from mineralforge.data_processing import preprocess_events
from mineralforge.models import FEATURE_COLUMNS
from mineralforge.pipeline import RockBurstPipeline
from mineralforge.synthetic import generate_synthetic_events

try:
    import streamlit as st
except Exception:  # pragma: no cover - optional UI dependency
    st = None


def run_dashboard() -> None:
    if st is None:
        raise RuntimeError("Streamlit is not installed. Run `pip install streamlit`.")

    st.set_page_config(page_title="MineralForge", layout="wide")
    st.title("MineralForge Rock-Burst Risk Dashboard")

    uploaded = st.sidebar.file_uploader("Upload event CSV", type=["csv"])
    if uploaded:
        frame = pd.read_csv(uploaded)
    else:
        frame = generate_synthetic_events(rows=80)
        frame["zone"] = ["Stope 3" if i % 2 else "Drive 2" for i in range(len(frame))]

    processed = preprocess_events(frame)
    pipeline = RockBurstPipeline.demo()
    assessments = []
    for _, row in processed.iterrows():
        features = {column: float(row[column]) for column in FEATURE_COLUMNS if column in row}
        zone = str(row.get("zone", "Unknown Zone"))
        assessments.append(pipeline.assess(features, zone=zone).to_dict())

    result_frame = pd.DataFrame(
        {
            "zone": [item["zone"] for item in assessments],
            "probability": [item["probability"] for item in assessments],
            "risk_level": [item["risk_level"] for item in assessments],
            "top_driver": [item["drivers"][0]["feature"] for item in assessments],
        }
    )

    high_count = int((result_frame["risk_level"] == "HIGH").sum())
    medium_count = int((result_frame["risk_level"] == "MEDIUM").sum())
    average_risk = float(result_frame["probability"].mean())

    left, middle, right = st.columns(3)
    left.metric("High risk events", high_count)
    middle.metric("Medium risk events", medium_count)
    right.metric("Average probability", f"{average_risk:.1%}")

    st.subheader("Risk timeline")
    st.line_chart(result_frame["probability"])

    st.subheader("Event table")
    st.dataframe(result_frame, use_container_width=True)

    st.subheader("Feature importance proxy")
    driver_counts = result_frame["top_driver"].value_counts()
    st.bar_chart(driver_counts)


if __name__ == "__main__":
    run_dashboard()
