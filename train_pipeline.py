from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from mineralforge.geotech import estimate_ppv_mm_s, scaled_distance_cube_root, scaled_distance_square_root

try:
    import optuna
except Exception:  # pragma: no cover - optional optimization dependency
    optuna = None

try:
    import mlflow
except Exception:  # pragma: no cover - optional experiment tracking dependency
    mlflow = None


TARGET_COLUMN = "Vibration_Level"
DEFAULT_DATA_PATH = Path("data/ground_vibration_dataset.csv")
ARTIFACT_DIR = Path("artifacts")


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column {TARGET_COLUMN!r}, found: {list(df.columns)}")
    return add_engineering_features(add_time_features(df))


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Timestamp" in df.columns:
        timestamp = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["hour"] = timestamp.dt.hour
        df["dayofweek"] = timestamp.dt.dayofweek
        df["month"] = timestamp.dt.month
        df = df.drop(columns=["Timestamp"])
    return df


def add_engineering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add blast-vibration features used by geotechnical engineers."""

    df = df.copy()
    charge_column = "Charge_Weight(kg)"
    if {charge_column, "Burden(m)", "Spacing(m)"}.issubset(df.columns):
        effective_distance = np.sqrt(df["Burden(m)"] ** 2 + df["Spacing(m)"] ** 2).clip(lower=0.1)
        charge = df[charge_column].clip(lower=0.1)
        df["Effective_Distance(m)"] = effective_distance
        df["Scaled_Distance_Sqrt"] = [
            scaled_distance_square_root(float(weight), float(distance))
            for weight, distance in zip(charge, effective_distance)
        ]
        df["Scaled_Distance_Cuberoot"] = [
            scaled_distance_cube_root(float(weight), float(distance))
            for weight, distance in zip(charge, effective_distance)
        ]
        soil_values = df["Soil_Type"] if "Soil_Type" in df.columns else pd.Series(["rock"] * len(df), index=df.index)
        df["Estimated_PPV(mm/s)"] = [
            estimate_ppv_mm_s(float(weight), float(distance), soil_type=str(soil))
            for weight, distance, soil in zip(charge, effective_distance, soil_values)
        ]

    acceleration_columns = ["Acc_X(m/sÂ²)", "Acc_Y(m/sÂ²)", "Acc_Z(m/sÂ²)"]
    if set(acceleration_columns).issubset(df.columns):
        df["Acceleration_Resultant(m/sÂ²)"] = np.sqrt(sum(df[column] ** 2 for column in acceleration_columns))

    if {"PPV(mm/s)", "Frequency(Hz)"}.issubset(df.columns):
        df["PPV_Frequency_Product"] = df["PPV(mm/s)"] * df["Frequency(Hz)"]

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # Blast_ID is an event identifier, not a physical predictor. Keeping it would
    # encourage memorization and hurt future predictions on new blast IDs.
    if "Blast_ID" in X.columns:
        X = X.drop(columns=["Blast_ID"])

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

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


def train(csv_path: Path, output_dir: Path, tuner: str = "grid", optuna_trials: int = 40) -> dict:
    df = load_data(csv_path)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    if tuner == "optuna":
        best_pipeline, tuning_summary = run_optuna_search(X_train, y_train, optuna_trials)
    else:
        best_pipeline, tuning_summary = run_grid_search(X_train, y_train)

    predictions = best_pipeline.predict(X_test)

    report = classification_report(y_test, predictions, output_dict=True)
    labels = sorted(y.unique())
    high_recall = report.get("High", {}).get("recall")
    metrics = {
        "dataset_path": str(csv_path),
        "rows": int(len(df)),
        "features_used": list(X.columns),
        "target_column": TARGET_COLUMN,
        "class_counts": y.value_counts().to_dict(),
        "tuning_method": tuning_summary["method"],
        "tuning_trials": tuning_summary["trials"],
        "best_model_type": tuning_summary["model_type"],
        "best_params": tuning_summary["best_params"],
        "cv_best_f1_macro": float(tuning_summary["best_score"]),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "high_class_recall": float(high_recall) if high_recall is not None else None,
        "classification_report": report,
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, output_dir / "vibration_detection_pipeline.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_feature_importance(best_pipeline, output_dir)
    track_with_mlflow(metrics, output_dir)

    return metrics


def run_grid_search(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline, dict]:
    preprocessor = build_preprocessor(X_train)
    base_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )

    param_grid = [
        {
            "model": [RandomForestClassifier(random_state=42, class_weight="balanced")],
            "model__n_estimators": [150, 300],
            "model__max_depth": [None, 8, 14],
            "model__min_samples_leaf": [1, 3],
            "model__criterion": ["gini", "entropy"],
        },
        {
            "model": [LogisticRegression(max_iter=3000, class_weight="balanced")],
            "model__C": [0.1, 1.0, 10.0],
        },
        {
            "model": [SVC(class_weight="balanced", probability=True, random_state=42)],
            "model__C": [0.5, 1.0, 5.0],
            "model__kernel": ["rbf"],
            "model__gamma": ["scale", "auto"],
        },
    ]

    search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=cv_splits_for(y_train), shuffle=True, random_state=42),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, {
        "method": "GridSearchCV",
        "trials": int(len(search.cv_results_["params"])),
        "model_type": type(search.best_estimator_.named_steps["model"]).__name__,
        "best_params": {key: str(value) for key, value in search.best_params_.items()},
        "best_score": float(search.best_score_),
    }


def run_optuna_search(X_train: pd.DataFrame, y_train: pd.Series, trials: int) -> tuple[Pipeline, dict]:
    if optuna is None:
        raise RuntimeError("Optuna is not installed. Run `pip install optuna` or use `--tuner grid`.")

    cv = StratifiedKFold(n_splits=cv_splits_for(y_train), shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 6, 10, 14, 20]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train)),
                ("model", RandomForestClassifier(random_state=42, class_weight="balanced", **params)),
            ]
        )
        scores = []
        for train_index, validation_index in cv.split(X_train, y_train):
            pipeline.fit(X_train.iloc[train_index], y_train.iloc[train_index])
            predictions = pipeline.predict(X_train.iloc[validation_index])
            report = classification_report(y_train.iloc[validation_index], predictions, output_dict=True, zero_division=0)
            macro_f1 = report["macro avg"]["f1-score"]
            high_recall = report.get("High", {}).get("recall", 0.0)
            scores.append(0.8 * macro_f1 + 0.2 * high_recall)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", study_name="mineralforge-rf-tuning")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    best_pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X_train)),
            ("model", RandomForestClassifier(random_state=42, class_weight="balanced", **study.best_params)),
        ]
    )
    best_pipeline.fit(X_train, y_train)
    return best_pipeline, {
        "method": "Optuna",
        "trials": int(len(study.trials)),
        "model_type": "RandomForestClassifier",
        "best_params": {key: str(value) for key, value in study.best_params.items()},
        "best_score": float(study.best_value),
    }


def cv_splits_for(y: pd.Series, desired_splits: int = 5) -> int:
    smallest_class = int(y.value_counts().min())
    if smallest_class < 2:
        raise ValueError("Each class needs at least two records for stratified cross-validation.")
    return min(desired_splits, smallest_class)


def save_feature_importance(pipeline: Pipeline, output_dir: Path) -> None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)


def track_with_mlflow(metrics: dict, output_dir: Path) -> None:
    if mlflow is None:
        return
    mlflow.set_tracking_uri(f"file:{(output_dir / 'mlruns').resolve()}")
    mlflow.set_experiment("mineralforge-vibration-risk")
    with mlflow.start_run():
        mlflow.log_metric("cv_best_f1_macro", metrics["cv_best_f1_macro"])
        mlflow.log_metric("test_accuracy", metrics["test_accuracy"])
        mlflow.log_params(metrics["best_params"])
        for artifact_name in ["metrics.json", "feature_importance.csv"]:
            artifact_path = output_dir / artifact_name
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a vibration seismic detection pipeline.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to input CSV.")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR, help="Directory for model and metrics.")
    parser.add_argument("--tuner", choices=["grid", "optuna"], default="grid", help="Hyperparameter search strategy.")
    parser.add_argument("--optuna-trials", type=int, default=40, help="Number of Optuna trials when --tuner optuna.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train(args.data, args.output_dir, tuner=args.tuner, optuna_trials=args.optuna_trials)
    print(f"Saved model to: {args.output_dir / 'vibration_detection_pipeline.joblib'}")
    print(f"Saved metrics to: {args.output_dir / 'metrics.json'}")
    print(f"Saved feature importance to: {args.output_dir / 'feature_importance.csv'}")
    print(f"Tuning method: {metrics['tuning_method']}")
    print(f"Best model type: {metrics['best_model_type']}")
    print(f"Best CV macro F1: {metrics['cv_best_f1_macro']:.3f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    if metrics["high_class_recall"] is not None:
        print(f"High-class recall: {metrics['high_class_recall']:.3f}")


if __name__ == "__main__":
    main()
