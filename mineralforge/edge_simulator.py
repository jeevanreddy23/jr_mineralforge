"""Raspberry Pi style edge loop simulator."""

from __future__ import annotations

import math

import numpy as np

from mineralforge.features import build_feature_frame
from mineralforge.pipeline import RockBurstPipeline


def synthetic_sensor_window(
    seconds: float = 4.0,
    sample_rate_hz: int = 1000,
    stress_multiplier: float = 1.0,
    random_state: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    t = np.linspace(0, seconds, int(seconds * sample_rate_hz), endpoint=False)
    acoustic = (
        0.05 * np.sin(2 * math.pi * 185 * t)
        + 0.02 * rng.normal(size=t.size)
        + stress_multiplier * 0.04 * np.sin(2 * math.pi * 360 * t)
    )
    vibration = (
        0.9 * np.sin(2 * math.pi * 38 * t)
        + 0.25 * rng.normal(size=t.size)
        + stress_multiplier * 0.5 * np.sin(2 * math.pi * 62 * t)
    )
    recent_energies = rng.gamma(2.2, 40.0 * stress_multiplier, size=90)
    return acoustic, vibration, recent_energies


def run_demo_assessment(zone: str = "Stope 3", stress_multiplier: float = 2.6) -> dict:
    acoustic, vibration, recent_energies = synthetic_sensor_window(stress_multiplier=stress_multiplier)
    frame = build_feature_frame(
        acoustic_signal=acoustic,
        vibration_signal=vibration,
        acoustic_sample_rate_hz=1000,
        vibration_sample_rate_hz=1000,
        recent_event_energies=recent_energies,
    )
    pipeline = RockBurstPipeline.demo()
    return pipeline.assess(frame.as_dict(), zone=zone).to_dict()
