"""Batch prediction entrypoint for field CSV files."""

from __future__ import annotations

import argparse
import json

from mineralforge.data_processing import load_event_csv, preprocess_events
from mineralforge.models import FEATURE_COLUMNS
from mineralforge.pipeline import RockBurstPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict vibration/seismic risk from a CSV file")
    parser.add_argument("csv", help="Input CSV path")
    parser.add_argument("--zone-column", default="zone", help="Column containing the zone name")
    args = parser.parse_args()

    frame = preprocess_events(load_event_csv(args.csv))
    pipeline = RockBurstPipeline.demo()
    results = []
    for _, row in frame.iterrows():
        features = {column: float(row[column]) for column in FEATURE_COLUMNS if column in row}
        zone = str(row.get(args.zone_column, "Unknown Zone"))
        results.append(pipeline.assess(features, zone=zone).to_dict())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
