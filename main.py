"""MineralForge command line interface."""

from __future__ import annotations

import argparse
import json

from mineralforge.edge_simulator import run_demo_assessment
from mineralforge.models import train_risk_model
from mineralforge.synthetic import generate_synthetic_events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MineralForge edge-AI rock-burst risk pipeline",
    )
    parser.add_argument("--zone", default="Stope 3", help="Mine zone name for the TARP output")
    parser.add_argument(
        "--stress",
        type=float,
        default=2.6,
        help="Synthetic stress multiplier for the edge sensor demo",
    )
    parser.add_argument(
        "--train-report",
        action="store_true",
        help="Print the synthetic validation report for the baseline model",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    if args.train_report:
        frame = generate_synthetic_events()
        trained = train_risk_model(frame)
        print(trained.validation_report)
        return

    assessment = run_demo_assessment(zone=args.zone, stress_multiplier=args.stress)
    if args.json:
        print(json.dumps(assessment, indent=2))
        return

    print("MineralForge Edge Assessment")
    print("=" * 32)
    print(f"Zone: {assessment['zone']}")
    print(f"Risk: {assessment['risk_level']} ({assessment['probability']:.1%})")
    print("\nTop drivers:")
    for driver in assessment["drivers"]:
        print(f"- {driver['feature']}: {driver['value']:.3f}")
    print("\nTARP action:")
    print(assessment["tarp"]["action"])
    print(assessment["tarp"]["notification"])
    print(assessment["tarp"]["access_control"])


if __name__ == "__main__":
    main()
