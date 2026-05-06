"""Shared project paths."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
VISUALIZATION_DIR = PROJECT_ROOT / "visualizations"
DEFAULT_DATA_PATH = DATA_DIR / "ground_vibration_dataset.csv"
DEFAULT_MODEL_PATH = ARTIFACT_DIR / "vibration_detection_pipeline.joblib"
TARGET_COLUMN = "Vibration_Level"
