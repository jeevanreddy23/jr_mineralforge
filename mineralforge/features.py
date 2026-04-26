"""Signal features used by the edge rock-burst pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


EPSILON = 1e-12


@dataclass(frozen=True)
class FeatureFrame:
    """Single time-window feature record produced on the edge device."""

    acoustic_rms: float
    acoustic_peak_frequency_hz: float
    spectral_entropy: float
    acoustic_event_rate: float
    vibration_ppv: float
    vibration_dominant_frequency_hz: float
    vibration_energy: float
    cumulative_energy: float

    def as_dict(self) -> dict[str, float]:
        return {
            "acoustic_rms": self.acoustic_rms,
            "acoustic_peak_frequency_hz": self.acoustic_peak_frequency_hz,
            "spectral_entropy": self.spectral_entropy,
            "acoustic_event_rate": self.acoustic_event_rate,
            "vibration_ppv": self.vibration_ppv,
            "vibration_dominant_frequency_hz": self.vibration_dominant_frequency_hz,
            "vibration_energy": self.vibration_energy,
            "cumulative_energy": self.cumulative_energy,
        }


def _to_1d_array(signal: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(signal), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty 1D sequence")
    return values


def _positive_fft(signal: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    centered = signal - np.mean(signal)
    magnitudes = np.abs(np.fft.rfft(centered))
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate_hz)
    return frequencies, magnitudes


def _peak_frequency(signal: np.ndarray, sample_rate_hz: float) -> float:
    frequencies, magnitudes = _positive_fft(signal, sample_rate_hz)
    if magnitudes.size <= 1:
        return 0.0
    index = int(np.argmax(magnitudes[1:]) + 1)
    return float(frequencies[index])


def _spectral_entropy(signal: np.ndarray, sample_rate_hz: float) -> float:
    _, magnitudes = _positive_fft(signal, sample_rate_hz)
    power = magnitudes**2
    total_power = float(np.sum(power))
    if total_power <= EPSILON:
        return 0.0
    probabilities = power / total_power
    entropy = -np.sum(probabilities * np.log2(probabilities + EPSILON))
    max_entropy = np.log2(probabilities.size)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def extract_acoustic_features(
    signal: Iterable[float],
    sample_rate_hz: float,
    event_threshold: float = 0.12,
) -> dict[str, float]:
    """Extract MEMS microphone features for one window."""

    values = _to_1d_array(signal)
    duration_seconds = values.size / sample_rate_hz
    crossings = np.count_nonzero(np.abs(values) >= event_threshold)
    return {
        "acoustic_rms": float(np.sqrt(np.mean(values**2))),
        "acoustic_peak_frequency_hz": _peak_frequency(values, sample_rate_hz),
        "spectral_entropy": _spectral_entropy(values, sample_rate_hz),
        "acoustic_event_rate": float(crossings / max(duration_seconds, EPSILON)),
    }


def extract_vibration_features(
    signal: Iterable[float],
    sample_rate_hz: float,
    velocity_scale: float = 1.0,
) -> dict[str, float]:
    """Extract accelerometer vibration features for one window."""

    values = _to_1d_array(signal) * velocity_scale
    return {
        "vibration_ppv": float(np.max(np.abs(values))),
        "vibration_dominant_frequency_hz": _peak_frequency(values, sample_rate_hz),
        "vibration_energy": float(np.sum(values**2)),
    }


def cumulative_energy(energies: Iterable[float]) -> float:
    """Return E_cum = sum(E_i), the key rock-burst precursor feature."""

    values = _to_1d_array(energies)
    return float(np.sum(values))


def build_feature_frame(
    acoustic_signal: Iterable[float],
    vibration_signal: Iterable[float],
    acoustic_sample_rate_hz: float,
    vibration_sample_rate_hz: float,
    recent_event_energies: Iterable[float],
) -> FeatureFrame:
    acoustic = extract_acoustic_features(acoustic_signal, acoustic_sample_rate_hz)
    vibration = extract_vibration_features(vibration_signal, vibration_sample_rate_hz)
    return FeatureFrame(
        acoustic_rms=acoustic["acoustic_rms"],
        acoustic_peak_frequency_hz=acoustic["acoustic_peak_frequency_hz"],
        spectral_entropy=acoustic["spectral_entropy"],
        acoustic_event_rate=acoustic["acoustic_event_rate"],
        vibration_ppv=vibration["vibration_ppv"],
        vibration_dominant_frequency_hz=vibration["vibration_dominant_frequency_hz"],
        vibration_energy=vibration["vibration_energy"],
        cumulative_energy=cumulative_energy(recent_event_energies),
    )
