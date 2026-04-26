"""End-to-end rock-burst risk assessment pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from mineralforge.models import TrainedRiskModel, explain_prediction, train_risk_model
from mineralforge.synthetic import generate_synthetic_events
from mineralforge.tarp import TarpAction, map_risk_to_tarp


@dataclass(frozen=True)
class RiskAssessment:
    zone: str
    probability: float
    risk_level: str
    drivers: list[dict[str, float]]
    tarp: TarpAction

    def to_dict(self) -> dict:
        return {
            "zone": self.zone,
            "probability": self.probability,
            "risk_level": self.risk_level,
            "drivers": self.drivers,
            "tarp": self.tarp.__dict__,
        }


class RockBurstPipeline:
    """Deployable risk pipeline: features, model, explanation, TARP action."""

    def __init__(self, model: TrainedRiskModel):
        self.model = model

    @classmethod
    def demo(cls, random_state: int = 42) -> "RockBurstPipeline":
        training_frame = generate_synthetic_events(random_state=random_state)
        return cls(train_risk_model(training_frame, random_state=random_state))

    def assess(self, features: dict[str, float], zone: str = "Stope 3") -> RiskAssessment:
        probability = self.model.predict_proba(features)
        risk_level = _risk_level(probability)
        drivers = explain_prediction(self.model, features, top_n=3)
        return RiskAssessment(
            zone=zone,
            probability=probability,
            risk_level=risk_level,
            drivers=drivers,
            tarp=map_risk_to_tarp(risk_level, zone),
        )


def _risk_level(probability: float) -> str:
    if probability >= 0.72:
        return "HIGH"
    if probability >= 0.38:
        return "MEDIUM"
    return "LOW"
