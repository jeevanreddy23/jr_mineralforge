"""FFT utilities for waveform safety analysis."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def spectrum(signal: Iterable[float], sample_rate_hz: float) -> dict[str, list[float]]:
    values = np.asarray(list(signal), dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("signal must contain at least two samples")
    centered = values - np.mean(values)
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate_hz)
    amplitudes = np.abs(np.fft.rfft(centered))
    return {
        "frequency_hz": frequencies.tolist(),
        "amplitude": amplitudes.tolist(),
    }


def dominant_frequency(signal: Iterable[float], sample_rate_hz: float) -> float:
    result = spectrum(signal, sample_rate_hz)
    amplitudes = np.asarray(result["amplitude"])
    frequencies = np.asarray(result["frequency_hz"])
    if amplitudes.size <= 1:
        return 0.0
    index = int(np.argmax(amplitudes[1:]) + 1)
    return float(frequencies[index])


def frequency_band_energy(
    signal: Iterable[float],
    sample_rate_hz: float,
    low_hz: float,
    high_hz: float,
) -> float:
    result = spectrum(signal, sample_rate_hz)
    frequencies = np.asarray(result["frequency_hz"])
    amplitudes = np.asarray(result["amplitude"])
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    return float(np.sum(amplitudes[mask] ** 2))
