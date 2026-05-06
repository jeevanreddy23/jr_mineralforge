"""Trigger Action Response Plan recommendations for blast vibration risk."""

from __future__ import annotations


def tarp_recommendation(risk_level: str) -> str:
    level = risk_level.strip().lower()
    if level == "high":
        return (
            "High Risk: Stop nearby non-essential activity, notify the geotechnical engineer, "
            "inspect exposed structures, review blast exclusion controls, and do not resume until cleared."
        )
    if level == "medium":
        return (
            "Medium Risk: Increase monitoring, verify sensor readings, review blast timing and charge controls, "
            "and inspect sensitive areas during the next field round."
        )
    return (
        "Low Risk: Continue standard monitoring and retain the event in the vibration log for trend analysis."
    )
