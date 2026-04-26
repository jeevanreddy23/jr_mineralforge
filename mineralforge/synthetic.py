"""Synthetic microseismic proxy data for demos and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mineralforge.models import FEATURE_COLUMNS


def generate_synthetic_events(rows: int = 600, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    baseline_energy = rng.gamma(shape=2.4, scale=35.0, size=rows)
    cumulative_energy = np.cumsum(baseline_energy) / rows
    event_bursts = rng.poisson(lam=2.0, size=rows)
    acoustic_rms = rng.normal(0.08, 0.02, size=rows).clip(0.005)
    vibration_ppv = rng.normal(1.8, 0.45, size=rows).clip(0.1)

    burst_index = rng.choice(rows, size=max(12, rows // 14), replace=False)
    cumulative_energy[burst_index] *= rng.uniform(2.5, 4.5, size=burst_index.size)
    event_bursts[burst_index] += rng.integers(5, 15, size=burst_index.size)
    acoustic_rms[burst_index] *= rng.uniform(1.7, 3.2, size=burst_index.size)
    vibration_ppv[burst_index] *= rng.uniform(1.6, 2.8, size=burst_index.size)

    frame = pd.DataFrame(
        {
            "acoustic_rms": acoustic_rms,
            "acoustic_peak_frequency_hz": rng.normal(185, 42, size=rows).clip(20),
            "spectral_entropy": rng.beta(3, 5, size=rows),
            "acoustic_event_rate": event_bursts.astype(float),
            "vibration_ppv": vibration_ppv,
            "vibration_dominant_frequency_hz": rng.normal(42, 11, size=rows).clip(2),
            "vibration_energy": baseline_energy * rng.uniform(0.8, 1.4, size=rows),
            "cumulative_energy": cumulative_energy,
        }
    )

    risk_score = (
        0.42 * _scale(frame["cumulative_energy"])
        + 0.22 * _scale(frame["acoustic_event_rate"])
        + 0.18 * _scale(frame["vibration_ppv"])
        + 0.12 * _scale(frame["vibration_energy"])
        + 0.06 * _scale(frame["acoustic_rms"])
    )
    frame["risk_event"] = (risk_score > np.quantile(risk_score, 0.84)).astype(int)
    return frame[[*FEATURE_COLUMNS, "risk_event"]]


def _scale(series: pd.Series) -> pd.Series:
    denominator = series.max() - series.min()
    if denominator == 0:
        return series * 0
    return (series - series.min()) / denominator
