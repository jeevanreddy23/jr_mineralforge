import numpy as np

from mineralforge.features import build_feature_frame, cumulative_energy, extract_acoustic_features


def test_cumulative_energy_sums_event_energy():
    assert cumulative_energy([2.5, 3.0, 4.5]) == 10.0


def test_acoustic_peak_frequency_detects_signal():
    sample_rate = 1000
    t = np.linspace(0, 1, sample_rate, endpoint=False)
    signal = np.sin(2 * np.pi * 120 * t)
    features = extract_acoustic_features(signal, sample_rate_hz=sample_rate)
    assert abs(features["acoustic_peak_frequency_hz"] - 120) <= 1


def test_feature_frame_contains_core_predictor():
    acoustic = np.ones(100)
    vibration = np.ones(100) * 0.5
    frame = build_feature_frame(acoustic, vibration, 100, 100, [1, 2, 3])
    assert frame.cumulative_energy == 6.0
    assert "cumulative_energy" in frame.as_dict()
