from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mineralforge.features.engineering import add_blast_engineering_features as add_engineering_features
from mineralforge.features.engineering import add_time_features
from mineralforge.models.training import (
    cv_splits_for,
    run_grid_search,
    run_optuna_search,
    train,
)
from mineralforge.utils.paths import ARTIFACT_DIR, DEFAULT_DATA_PATH, TARGET_COLUMN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MineralForge blast vibration risk predictor.")
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
    print(f"High-risk recall: {metrics['high_risk_recall']:.3f}")


if __name__ == "__main__":
    main()
