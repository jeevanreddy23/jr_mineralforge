"""Data loading and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mineralforge.features.engineering import prepare_training_frame
from mineralforge.utils.paths import TARGET_COLUMN


def load_dataset(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Expected target column {TARGET_COLUMN!r}, found: {list(frame.columns)}")
    return prepare_training_frame(frame)


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = frame[TARGET_COLUMN]
    x = frame.drop(columns=[TARGET_COLUMN])
    if "Blast_ID" in x.columns:
        x = x.drop(columns=["Blast_ID"])
    return x, y


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
