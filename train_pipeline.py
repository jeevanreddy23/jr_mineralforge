from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
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


TARGET_COLUMN = "Vibration_Level"
DEFAULT_DATA_PATH = Path("data/ground_vibration_dataset.csv")
ARTIFACT_DIR = Path("artifacts")


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column {TARGET_COLUMN!r}, found: {list(df.columns)}")
    return add_time_features(df)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Timestamp" in df.columns:
        timestamp = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["hour"] = timestamp.dt.hour
        df["dayofweek"] = timestamp.dt.dayofweek
        df["month"] = timestamp.dt.month
        df = df.drop(columns=["Timestamp"])
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


def train(csv_path: Path, output_dir: Path) -> dict:
    df = load_data(csv_path)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

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
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    predictions = best_pipeline.predict(X_test)

    report = classification_report(y_test, predictions, output_dict=True)
    labels = sorted(y.unique())
    metrics = {
        "dataset_path": str(csv_path),
        "rows": int(len(df)),
        "features_used": list(X.columns),
        "target_column": TARGET_COLUMN,
        "class_counts": y.value_counts().to_dict(),
        "best_params": {key: str(value) for key, value in search.best_params_.items()},
        "cv_best_f1_macro": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
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

    return metrics


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a vibration seismic detection pipeline.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to input CSV.")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR, help="Directory for model and metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train(args.data, args.output_dir)
    print(f"Saved model to: {args.output_dir / 'vibration_detection_pipeline.joblib'}")
    print(f"Saved metrics to: {args.output_dir / 'metrics.json'}")
    print(f"Saved feature importance to: {args.output_dir / 'feature_importance.csv'}")
    print(f"Best CV macro F1: {metrics['cv_best_f1_macro']:.3f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")


if __name__ == "__main__":
    main()
