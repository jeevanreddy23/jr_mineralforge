from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from train_pipeline import TARGET_COLUMN, add_time_features


DEFAULT_MODEL_PATH = Path("artifacts/vibration_detection_pipeline.joblib")


def prepare_prediction_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = add_time_features(df)
    columns_to_drop = [column for column in [TARGET_COLUMN, "Blast_ID"] if column in df.columns]
    return df.drop(columns=columns_to_drop)


def predict(model_path: Path, input_csv: Path, output_csv: Path) -> pd.DataFrame:
    model = joblib.load(model_path)
    X = prepare_prediction_data(input_csv)
    predictions = model.predict(X)

    output = pd.read_csv(input_csv)
    output["Predicted_Vibration_Level"] = predictions

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        for label, probability in zip(model.classes_, probabilities.T):
            output[f"Probability_{label}"] = probability

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict vibration levels from a trained pipeline.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to trained .joblib model.")
    parser.add_argument("--input", type=Path, required=True, help="CSV file to score.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/predictions.csv"), help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = predict(args.model, args.input, args.output)
    print(f"Saved {len(output)} predictions to: {args.output}")


if __name__ == "__main__":
    main()
