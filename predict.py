from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mineralforge.models.prediction import predict_file, prepare_prediction_data
from mineralforge.utils.paths import DEFAULT_MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Low, Medium, or High blast vibration risk.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to trained .joblib model.")
    parser.add_argument("--input", type=Path, required=True, help="CSV file to score.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/predictions.csv"), help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = predict_file(args.model, args.input, args.output)
    print(f"Saved {len(output)} blast vibration predictions to: {args.output}")


if __name__ == "__main__":
    main()
