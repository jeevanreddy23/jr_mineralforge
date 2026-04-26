"""CSV preprocessing for field blast and vibration records."""

from __future__ import annotations

import pandas as pd

from mineralforge.geotech import BlastContext, blast_feature_dict


SOIL_COLUMNS = ["soil_type_clay", "soil_type_fill", "soil_type_fractured_rock", "soil_type_rock", "soil_type_sand"]


def load_event_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_events(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "timestamp" in data.columns:
        timestamp = pd.to_datetime(data["timestamp"], errors="coerce")
        data["hour"] = timestamp.dt.hour.fillna(0).astype(int)
        data["dayofweek"] = timestamp.dt.dayofweek.fillna(0).astype(int)

    if "soil_type" not in data.columns:
        data["soil_type"] = "rock"
    data["soil_type"] = data["soil_type"].fillna("rock").astype(str).str.lower()

    if {"charge_mass_kg", "distance_m"}.issubset(data.columns):
        geotech_rows = [
            blast_feature_dict(
                BlastContext(
                    charge_mass_kg=float(row["charge_mass_kg"]),
                    distance_m=float(row["distance_m"]),
                    soil_type=str(row.get("soil_type", "rock")),
                    structure_distance_m=float(row["structure_distance_m"])
                    if "structure_distance_m" in data.columns and pd.notna(row.get("structure_distance_m"))
                    else None,
                )
            )
            for _, row in data.iterrows()
        ]
        geotech = pd.DataFrame(geotech_rows, index=data.index)
        for column in geotech.columns:
            data[column] = geotech[column]

    encoded = pd.get_dummies(data["soil_type"], prefix="soil_type")
    data = pd.concat([data, encoded], axis=1)
    for column in SOIL_COLUMNS:
        if column not in data.columns:
            data[column] = 0
    return data


def numeric_feature_frame(frame: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    excluded = {"timestamp", "soil_type"}
    if target_column:
        excluded.add(target_column)
    return frame[[column for column in frame.columns if column not in excluded]].select_dtypes("number")
