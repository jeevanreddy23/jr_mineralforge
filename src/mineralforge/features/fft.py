"""FFT helpers for blast vibration waveforms."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def dominant_frequency(signal: Iterable[float], sample_rate_hz: float) -> float:
    values = np.asarray(list(signal), dtype=float)
    if values.size < 2:
        raise ValueError("signal must contain at least two samples")
    centered = values - np.mean(values)
    amplitudes = np.abs(np.fft.rfft(centered))
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate_hz)
    if amplitudes.size <= 1:
        return 0.0
    return float(frequencies[int(np.argmax(amplitudes[1:]) + 1)])


def frequency_band_energy(signal: Iterable[float], sample_rate_hz: float, low_hz: float, high_hz: float) -> float:
    values = np.asarray(list(signal), dtype=float)
    if values.size < 2:
        raise ValueError("signal must contain at least two samples")
    centered = values - np.mean(values)
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate_hz)
    amplitudes = np.abs(np.fft.rfft(centered))
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    return float(np.sum(amplitudes[mask] ** 2))
