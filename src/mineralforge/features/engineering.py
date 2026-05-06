"""Feature engineering for blast vibration risk prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mineralforge.geotech.calculations import (
    effective_distance_m,
    estimate_ppv_mm_s,
    ppv_frequency_product,
    scaled_distance_cube_root,
    scaled_distance_square_root,
)


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "Timestamp" in data.columns:
        timestamp = pd.to_datetime(data["Timestamp"], errors="coerce")
        data["hour"] = timestamp.dt.hour
        data["dayofweek"] = timestamp.dt.dayofweek
        data["month"] = timestamp.dt.month
        data = data.drop(columns=["Timestamp"])
    return data


def add_blast_engineering_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    charge_column = "Charge_Weight(kg)"
    if {charge_column, "Burden(m)", "Spacing(m)"}.issubset(data.columns):
        distance = [
            effective_distance_m(float(burden), float(spacing))
            for burden, spacing in zip(data["Burden(m)"], data["Spacing(m)"])
        ]
        charge = data[charge_column].clip(lower=0.1)
        data["Effective_Distance(m)"] = distance
        data["Scaled_Distance_Sqrt"] = [
            scaled_distance_square_root(float(weight), float(dist))
            for weight, dist in zip(charge, distance)
        ]
        data["Scaled_Distance_Cuberoot"] = [
            scaled_distance_cube_root(float(weight), float(dist))
            for weight, dist in zip(charge, distance)
        ]
        soil_values = data["Soil_Type"] if "Soil_Type" in data.columns else pd.Series(["rock"] * len(data), index=data.index)
        data["Estimated_PPV(mm/s)"] = [
            estimate_ppv_mm_s(float(weight), float(dist), soil_type=str(soil))
            for weight, dist, soil in zip(charge, distance, soil_values)
        ]

    acceleration_columns = ["Acc_X(m/s²)", "Acc_Y(m/s²)", "Acc_Z(m/s²)"]
    fallback_columns = ["Acc_X(m/sÂ²)", "Acc_Y(m/sÂ²)", "Acc_Z(m/sÂ²)"]
    selected_columns = acceleration_columns if set(acceleration_columns).issubset(data.columns) else fallback_columns
    if set(selected_columns).issubset(data.columns):
        data["Acceleration_Resultant(m/s²)"] = np.sqrt(sum(data[column] ** 2 for column in selected_columns))

    if {"PPV(mm/s)", "Frequency(Hz)"}.issubset(data.columns):
        data["PPV_Frequency_Product"] = [
            ppv_frequency_product(float(ppv), float(freq))
            for ppv, freq in zip(data["PPV(mm/s)"], data["Frequency(Hz)"])
        ]
    return data


def prepare_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return add_blast_engineering_features(add_time_features(frame))
