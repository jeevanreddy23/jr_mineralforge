"""Model training and explanation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
except Exception:  # pragma: no cover - optional edge dependency
    IsolationForest = None
    RandomForestClassifier = None
    classification_report = None
    train_test_split = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None


FEATURE_COLUMNS = [
    "acoustic_rms",
    "acoustic_peak_frequency_hz",
    "spectral_entropy",
    "acoustic_event_rate",
    "vibration_ppv",
    "vibration_dominant_frequency_hz",
    "vibration_energy",
    "cumulative_energy",
]


@dataclass
class TrainedRiskModel:
    estimator: object
    feature_columns: list[str]
    validation_report: str

    def predict_proba(self, features: Mapping[str, float]) -> float:
        row = pd.DataFrame([{name: float(features[name]) for name in self.feature_columns}])
        probabilities = self.estimator.predict_proba(row)
        return float(probabilities[0, 1])


def _build_estimator(random_state: int) -> object:
    if XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=140,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )
    if RandomForestClassifier is not None:
        return RandomForestClassifier(
            n_estimators=180,
            max_depth=7,
            class_weight="balanced",
            random_state=random_state,
        )
    return EnergyThresholdClassifier()


class EnergyThresholdClassifier:
    """Small dependency-free classifier for Raspberry Pi demos and CI smoke tests."""

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "EnergyThresholdClassifier":
        scored_columns = [
            "cumulative_energy",
            "acoustic_event_rate",
            "vibration_ppv",
            "vibration_energy",
            "acoustic_rms",
        ]
        self.feature_min_ = x[scored_columns].min()
        self.feature_range_ = (x[scored_columns].max() - self.feature_min_).replace(0, 1.0)
        score = self._score_frame(x)
        candidates = np.quantile(score, np.linspace(0.05, 0.95, 91))
        labels = y.to_numpy(dtype=int)
        accuracies = [np.mean((score >= threshold).astype(int) == labels) for threshold in candidates]
        self.threshold_ = float(candidates[int(np.argmax(accuracies))])
        self.scale_ = float(max(np.std(score), 1e-6))
        self.feature_importances_ = np.array([0.05, 0.04, 0.05, 0.18, 0.13, 0.04, 0.14, 0.37])
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        score = (self._score_frame(x) - self.threshold_) / self.scale_
        probability = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - probability, probability])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    def _score_frame(self, x: pd.DataFrame) -> np.ndarray:
        return (
            0.42 * self._scale_feature(x, "cumulative_energy")
            + 0.22 * self._scale_feature(x, "acoustic_event_rate")
            + 0.18 * self._scale_feature(x, "vibration_ppv")
            + 0.12 * self._scale_feature(x, "vibration_energy")
            + 0.06 * self._scale_feature(x, "acoustic_rms")
        ).to_numpy(dtype=float)

    def _scale_feature(self, x: pd.DataFrame, feature: str) -> pd.Series:
        return (x[feature] - self.feature_min_[feature]) / self.feature_range_[feature]


def _fallback_split(
    x: pd.DataFrame,
    y: pd.Series,
    test_fraction: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(x))
    rng.shuffle(indices)
    split = max(1, int(len(indices) * (1.0 - test_fraction)))
    train_index = indices[:split]
    test_index = indices[split:]
    return x.iloc[train_index], x.iloc[test_index], y.iloc[train_index], y.iloc[test_index]


def _report(y_true: pd.Series, y_pred: np.ndarray) -> str:
    if classification_report is not None:
        return classification_report(y_true, y_pred, zero_division=0)
    accuracy = float(np.mean(y_true.to_numpy() == y_pred))
    return f"accuracy: {accuracy:.3f}\nbackend: EnergyThresholdClassifier"


def train_risk_model(
    frame: pd.DataFrame,
    target_column: str = "risk_event",
    random_state: int = 42,
) -> TrainedRiskModel:
    missing = [column for column in [*FEATURE_COLUMNS, target_column] if column not in frame.columns]
    if missing:
        raise ValueError(f"training data is missing required columns: {missing}")

    x = frame[FEATURE_COLUMNS]
    y = frame[target_column].astype(int)
    if train_test_split is not None:
        stratify = y if y.nunique() > 1 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=random_state,
            stratify=stratify,
        )
    else:
        x_train, x_test, y_train, y_test = _fallback_split(x, y, random_state=random_state)
    estimator = _build_estimator(random_state)
    estimator.fit(x_train, y_train)
    predictions = estimator.predict(x_test)
    report = _report(y_test, predictions)
    return TrainedRiskModel(estimator=estimator, feature_columns=FEATURE_COLUMNS.copy(), validation_report=report)


def train_anomaly_detector(frame: pd.DataFrame, random_state: int = 42) -> IsolationForest:
    if IsolationForest is None:
        raise RuntimeError("scikit-learn is required for IsolationForest anomaly detection")
    detector = IsolationForest(contamination=0.08, random_state=random_state)
    detector.fit(frame[FEATURE_COLUMNS])
    return detector


def explain_prediction(model: TrainedRiskModel, features: Mapping[str, float], top_n: int = 3) -> list[dict[str, float]]:
    row = pd.DataFrame([{name: float(features[name]) for name in model.feature_columns}])
    if shap is not None:
        explainer = shap.TreeExplainer(model.estimator)
        values = explainer.shap_values(row)
        if isinstance(values, list):
            values = values[-1]
        contributions = np.asarray(values).reshape(-1)
    elif hasattr(model.estimator, "feature_importances_"):
        contributions = np.asarray(model.estimator.feature_importances_) * row.iloc[0].to_numpy()
    else:
        contributions = row.iloc[0].to_numpy()

    ranked = sorted(
        (
            {
                "feature": feature,
                "value": float(row.iloc[0][feature]),
                "contribution": float(contribution),
            }
            for feature, contribution in zip(model.feature_columns, contributions)
        ),
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )
    return ranked[:top_n]
