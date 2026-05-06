"""Model training for blast vibration risk prediction."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from mineralforge.data.processing import build_preprocessor, load_dataset, split_features_target
from mineralforge.utils.paths import ARTIFACT_DIR, DEFAULT_DATA_PATH, TARGET_COLUMN

try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None

try:
    import shap
except Exception:  # pragma: no cover
    shap = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


class XGBoostRiskClassifier(BaseEstimator, ClassifierMixin):
    """Small sklearn-compatible wrapper that supports string risk labels."""

    def __init__(self, n_estimators: int = 150, max_depth: int = 3, learning_rate: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, x, y):
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed")
        self.encoder_ = LabelEncoder()
        y_encoded = self.encoder_.fit_transform(y)
        self.classes_ = self.encoder_.classes_
        self.model_ = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        self.model_.fit(x, y_encoded)
        if hasattr(self.model_, "feature_importances_"):
            self.feature_importances_ = self.model_.feature_importances_
        return self

    def predict(self, x):
        return self.encoder_.inverse_transform(self.model_.predict(x).astype(int))

    def predict_proba(self, x):
        return self.model_.predict_proba(x)

    def get_params(self, deep: bool = True):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def train(csv_path: Path = DEFAULT_DATA_PATH, output_dir: Path = ARTIFACT_DIR, tuner: str = "grid", optuna_trials: int = 40) -> dict:
    frame = load_dataset(csv_path)
    x, y = split_features_target(frame)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    x_train_for_fit = x_train
    y_train_for_fit = y_train
    prefit_preprocessor = build_preprocessor(x_train)

    if tuner == "optuna":
        best_pipeline, tuning_summary = run_optuna_search(x_train, y_train, optuna_trials)
    else:
        best_pipeline, tuning_summary = run_grid_search(x_train_for_fit, y_train_for_fit, prefit_preprocessor)

    predictions = best_pipeline.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    labels = sorted(y.unique())
    metrics = {
        "dataset_path": str(csv_path),
        "rows": int(len(frame)),
        "use_case": "Blast vibration risk predictor",
        "input_features": ["Charge_Weight(kg)", "Burden(m)", "Spacing(m)", "PPV(mm/s)", "Frequency(Hz)", "Soil_Type"],
        "target_column": TARGET_COLUMN,
        "class_counts": y.value_counts().to_dict(),
        "tuning_method": tuning_summary["method"],
        "tuning_trials": tuning_summary["trials"],
        "best_model_type": tuning_summary["model_type"],
        "best_params": tuning_summary["best_params"],
        "cv_best_f1_macro": float(tuning_summary["best_score"]),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "high_risk_recall": float(report.get("High", {}).get("recall", 0.0)),
        "classification_report": report,
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(y_test, predictions, labels=labels).tolist()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, output_dir / "vibration_detection_pipeline.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_feature_importance(best_pipeline, output_dir)
    save_shap_explanation(best_pipeline, x_test, output_dir)
    save_model_card(metrics, output_dir)
    return metrics


def run_grid_search(x_train: pd.DataFrame, y_train: pd.Series, preprocessor) -> tuple[Pipeline, dict]:
    base_pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", RandomForestClassifier(random_state=42))])
    param_grid = [
        {
            "model": [RandomForestClassifier(random_state=42, class_weight="balanced")],
            "model__n_estimators": [150, 300],
            "model__max_depth": [None, 8, 14],
            "model__min_samples_leaf": [1, 3],
            "model__criterion": ["gini", "entropy"],
        },
        {"model": [LogisticRegression(max_iter=3000, class_weight="balanced")], "model__C": [0.1, 1.0, 10.0]},
        {
            "model": [SVC(class_weight="balanced", probability=True, random_state=42)],
            "model__C": [0.5, 1.0, 5.0],
            "model__kernel": ["rbf"],
            "model__gamma": ["scale", "auto"],
        },
    ]
    if XGBClassifier is not None:
        param_grid.append(
            {
                "model": [
                    XGBoostRiskClassifier(random_state=42)
                ],
                "model__n_estimators": [150, 300],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
            }
        )
    search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=cv_splits_for(y_train), shuffle=True, random_state=42),
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, {
        "method": "GridSearchCV",
        "trials": int(len(search.cv_results_["params"])),
        "model_type": type(search.best_estimator_.named_steps["model"]).__name__,
        "best_params": {key: str(value) for key, value in search.best_params_.items()},
        "best_score": float(search.best_score_),
    }


def run_optuna_search(x_train: pd.DataFrame, y_train: pd.Series, trials: int) -> tuple[Pipeline, dict]:
    if optuna is None:
        raise RuntimeError("Optuna is not installed. Use --tuner grid or install optuna.")
    cv = StratifiedKFold(n_splits=cv_splits_for(y_train), shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 6, 10, 14, 20]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        }
        scores = []
        for train_index, validation_index in cv.split(x_train, y_train):
            pipeline = Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(x_train)),
                    ("model", RandomForestClassifier(random_state=42, class_weight="balanced", **params)),
                ]
            )
            pipeline.fit(x_train.iloc[train_index], y_train.iloc[train_index])
            prediction = pipeline.predict(x_train.iloc[validation_index])
            report = classification_report(y_train.iloc[validation_index], prediction, output_dict=True, zero_division=0)
            scores.append(0.8 * report["macro avg"]["f1-score"] + 0.2 * report.get("High", {}).get("recall", 0.0))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", study_name="mineralforge-blast-vibration")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(x_train)),
            ("model", RandomForestClassifier(random_state=42, class_weight="balanced", **study.best_params)),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline, {
        "method": "Optuna",
        "trials": int(len(study.trials)),
        "model_type": "RandomForestClassifier",
        "best_params": {key: str(value) for key, value in study.best_params.items()},
        "best_score": float(study.best_value),
    }


def save_feature_importance(pipeline: Pipeline, output_dir: Path) -> None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importance = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)


def save_shap_explanation(pipeline: Pipeline, sample: pd.DataFrame, output_dir: Path) -> None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    transformed = pipeline.named_steps["preprocess"].transform(sample)
    try:
        if shap is None:
            raise RuntimeError("SHAP is not installed")
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            absolute_values = np.mean([np.abs(class_values) for class_values in values], axis=(0, 1))
        else:
            absolute_values = np.mean(np.abs(values), axis=0)
        explanation = pd.DataFrame({"feature": feature_names, "mean_abs_shap": absolute_values}).sort_values("mean_abs_shap", ascending=False)
        explanation.to_csv(output_dir / "shap_summary.csv", index=False)
    except Exception:
        explanation = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": model.feature_importances_,
                "note": "Fallback to model feature_importances_ because SHAP was unavailable for this run.",
            }
        ).sort_values("mean_abs_shap", ascending=False)
        explanation.to_csv(output_dir / "shap_summary.csv", index=False)


def save_model_card(metrics: dict, output_dir: Path) -> None:
    model_card = f"""# Model Card: Blast Vibration Risk Predictor

## Intended Use
Predict Low, Medium, or High blast vibration risk from charge weight, burden, spacing, PPV, frequency, and soil/rock type.

## Validation Status
This model is trained on the included CSV dataset. It is an MVP and is not validated for production mine safety decisions.

## Key Metrics
- Macro F1: {metrics['cv_best_f1_macro']:.3f}
- Test Accuracy: {metrics['test_accuracy']:.3f}
- High Risk Recall: {metrics['high_risk_recall']:.3f}

## Output
The model outputs a vibration risk class. The application also reports estimated PPV and an explanation/recommendation layer.
"""
    (output_dir / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")


def cv_splits_for(y: pd.Series, desired_splits: int = 5) -> int:
    smallest_class = int(y.value_counts().min())
    if smallest_class < 2:
        raise ValueError("Each class needs at least two records for stratified cross-validation.")
    return min(desired_splits, smallest_class)
