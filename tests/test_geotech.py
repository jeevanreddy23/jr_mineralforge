from mineralforge.fft import dominant_frequency, frequency_band_energy
from mineralforge.geotech import estimate_ppv_mm_s, scaled_distance_cube_root, scaled_distance_square_root
import numpy as np


def test_scaled_distance_calculations():
    assert round(scaled_distance_square_root(100, 50), 2) == 5.0
    assert round(scaled_distance_cube_root(125, 50), 2) == 10.0


def test_estimated_ppv_decreases_with_distance():
    near = estimate_ppv_mm_s(100, 40)
    far = estimate_ppv_mm_s(100, 120)
    assert near > far


def test_fft_detects_dominant_frequency_and_band_energy():
    sample_rate = 1000
    t = np.linspace(0, 1, sample_rate, endpoint=False)
    signal = np.sin(2 * np.pi * 50 * t)
    assert abs(dominant_frequency(signal, sample_rate) - 50) <= 1
    assert frequency_band_energy(signal, sample_rate, 45, 55) > 0
