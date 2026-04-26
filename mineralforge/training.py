"""Production-oriented model training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mineralforge.data_processing import numeric_feature_frame, preprocess_events
from mineralforge.models import EnergyThresholdClassifier

try:
    from imblearn.over_sampling import SMOTE
except Exception:  # pragma: no cover - optional dependency
    SMOTE = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GridSearchCV, train_test_split
except Exception:  # pragma: no cover - optional dependency
    RandomForestClassifier = None
    GridSearchCV = None
    f1_score = None
    train_test_split = None

try:
    import mlflow
except Exception:  # pragma: no cover - optional dependency
    mlflow = None


@dataclass(frozen=True)
class TrainingResult:
    estimator: object
    feature_columns: list[str]
    best_params: dict
    f1_score: float
    tracking_uri: str | None = None


def train_from_frame(
    frame: pd.DataFrame,
    target_column: str = "risk_event",
    use_smote: bool = True,
    track_mlflow: bool = False,
    artifact_dir: str = "artifacts",
    random_state: int = 42,
) -> TrainingResult:
    processed = preprocess_events(frame)
    x = numeric_feature_frame(processed, target_column=target_column)
    y = processed[target_column].astype(int)

    if train_test_split is None or RandomForestClassifier is None:
        estimator = EnergyThresholdClassifier().fit(x, y)
        predictions = estimator.predict(x)
        score = float(np.mean(predictions == y.to_numpy()))
        return TrainingResult(estimator, list(x.columns), {"backend": "EnergyThresholdClassifier"}, score)

    stratify = y if y.nunique() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=stratify,
    )

    if use_smote and SMOTE is not None and y_train.value_counts().min() >= 2:
        k_neighbors = min(5, int(y_train.value_counts().min()) - 1)
        x_train, y_train = SMOTE(random_state=random_state, k_neighbors=k_neighbors).fit_resample(x_train, y_train)

    estimator = RandomForestClassifier(class_weight="balanced", random_state=random_state)
    params = {
        "n_estimators": [120, 240],
        "max_depth": [4, 8, None],
        "min_samples_leaf": [1, 3],
    }
    if GridSearchCV is not None:
        search = GridSearchCV(estimator, params, scoring="f1", cv=3, n_jobs=-1)
        search.fit(x_train, y_train)
        estimator = search.best_estimator_
        best_params = search.best_params_
    else:
        estimator.fit(x_train, y_train)
        best_params = {"class_weight": "balanced"}

    predictions = estimator.predict(x_test)
    score = float(f1_score(y_test, predictions, zero_division=0)) if f1_score else float(np.mean(predictions == y_test))
    tracking_uri = _track_run(track_mlflow, artifact_dir, best_params, score, list(x.columns))
    return TrainingResult(estimator, list(x.columns), best_params, score, tracking_uri)


def _track_run(
    enabled: bool,
    artifact_dir: str,
    params: dict,
    score: float,
    feature_columns: list[str],
) -> str | None:
    if not enabled or mlflow is None:
        return None
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    tracking_uri = f"file:{Path(artifact_dir, 'mlruns').resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("mineralforge-risk")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("f1_score", score)
        feature_path = Path(artifact_dir, "feature_columns.txt")
        feature_path.write_text("\n".join(feature_columns), encoding="utf-8")
        mlflow.log_artifact(str(feature_path))
    return tracking_uri
