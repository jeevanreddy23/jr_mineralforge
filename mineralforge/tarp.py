"""Trigger Action Response Plan mapping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TarpAction:
    risk_level: str
    action: str
    notification: str
    access_control: str


def map_risk_to_tarp(risk_level: str, zone: str) -> TarpAction:
    level = risk_level.upper()
    if level == "HIGH":
        return TarpAction(
            risk_level=level,
            action=f"Evacuate {zone} immediately and suspend work until geotechnical clearance.",
            notification="Notify geotechnical engineer, shift supervisor, and control room.",
            access_control="Restrict all non-essential access and establish exclusion barricades.",
        )
    if level == "MEDIUM":
        return TarpAction(
            risk_level=level,
            action=f"Increase monitoring frequency in {zone} and prepare crews for controlled withdrawal.",
            notification="Notify shift supervisor and geotechnical engineer for review.",
            access_control="Limit access to essential personnel and verify refuge/egress readiness.",
        )
    return TarpAction(
        risk_level="LOW",
        action=f"Continue standard operations in {zone} with routine monitoring.",
        notification="Log the assessment in the shift record.",
        access_control="No additional restriction required.",
    )
