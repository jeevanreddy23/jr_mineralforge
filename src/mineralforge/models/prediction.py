"""Prediction helpers for blast vibration risk."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from mineralforge.features.engineering import prepare_training_frame
from mineralforge.geotech.recommendations import tarp_recommendation
from mineralforge.utils.paths import DEFAULT_MODEL_PATH, TARGET_COLUMN


DEFAULT_MODEL_INPUTS = {
    "Delay(ms)": 0.0,
    "Temperature(°C)": 25.0,
    "Temperature(Â°C)": 25.0,
    "Wind_Speed(m/s)": 0.0,
    "Seismometer(m/s²)": 0.0,
    "Seismometer(m/sÂ²)": 0.0,
    "Geophone(mm/s)": 0.0,
    "Acc_X(m/s²)": 0.0,
    "Acc_Y(m/s²)": 0.0,
    "Acc_Z(m/s²)": 0.0,
    "Acc_X(m/sÂ²)": 0.0,
    "Acc_Y(m/sÂ²)": 0.0,
    "Acc_Z(m/sÂ²)": 0.0,
    "PSD_Value": 0.0,
    "Acceleration_Resultant(m/s²)": 0.0,
    "Acceleration_Resultant(m/sÂ²)": 0.0,
    "hour": 0,
    "dayofweek": 0,
    "month": 1,
}


def prepare_prediction_data(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame = prepare_training_frame(frame)
    for column, value in DEFAULT_MODEL_INPUTS.items():
        if column not in frame.columns:
            frame[column] = value
    drop_columns = [column for column in [TARGET_COLUMN, "Blast_ID"] if column in frame.columns]
    return frame.drop(columns=drop_columns)


def predict_file(model_path: Path, input_csv: Path, output_csv: Path) -> pd.DataFrame:
    model = joblib.load(model_path)
    x = prepare_prediction_data(input_csv)
    output = pd.read_csv(input_csv)
    predictions = model.predict(x)
    output["Predicted_Vibration_Level"] = predictions
    output["TARP_Recommendation"] = [tarp_recommendation(level) for level in predictions]
    if "Estimated_PPV(mm/s)" not in output.columns:
        engineered = prepare_training_frame(output)
        if "Estimated_PPV(mm/s)" in engineered.columns:
            output["Estimated_PPV(mm/s)"] = engineered["Estimated_PPV(mm/s)"]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        for label, probability in zip(model.classes_, probabilities.T):
            output[f"Probability_{label}"] = probability
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    return output


def predict_record(record: dict, model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    temp = pd.DataFrame([record])
    temp_path = Path("artifacts/.api_input.csv")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp.to_csv(temp_path, index=False)
    try:
        model = joblib.load(model_path)
        x = prepare_prediction_data(temp_path)
        risk = str(model.predict(x)[0])
        result = {"risk_level": risk, "tarp_recommendation": tarp_recommendation(risk)}
        engineered = prepare_training_frame(temp)
        if "Estimated_PPV(mm/s)" in engineered.columns:
            result["estimated_ppv_mm_s"] = float(engineered["Estimated_PPV(mm/s)"].iloc[0])
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(x)[0]
            result["probabilities"] = {str(label): float(prob) for label, prob in zip(model.classes_, probabilities)}
        return result
    finally:
        temp_path.unlink(missing_ok=True)
