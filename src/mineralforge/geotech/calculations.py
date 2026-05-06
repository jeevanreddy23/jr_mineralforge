"""Blast vibration engineering calculations."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


SOIL_ATTENUATION = {
    "hard": 0.92,
    "rock": 1.00,
    "medium": 1.10,
    "soft": 1.28,
    "fractured_rock": 1.18,
    "sand": 1.25,
    "clay": 1.35,
    "fill": 1.45,
}


@dataclass(frozen=True)
class BlastInputs:
    charge_weight_kg: float
    burden_m: float
    spacing_m: float
    ppv_mm_s: float | None = None
    frequency_hz: float | None = None
    soil_type: str = "rock"


def effective_distance_m(burden_m: float, spacing_m: float) -> float:
    _validate_positive(burden_m, "burden_m")
    _validate_positive(spacing_m, "spacing_m")
    return float((burden_m**2 + spacing_m**2) ** 0.5)


def scaled_distance_square_root(charge_weight_kg: float, distance_m: float) -> float:
    _validate_positive(charge_weight_kg, "charge_weight_kg")
    _validate_positive(distance_m, "distance_m")
    return float(distance_m / sqrt(charge_weight_kg))


def scaled_distance_cube_root(charge_weight_kg: float, distance_m: float) -> float:
    _validate_positive(charge_weight_kg, "charge_weight_kg")
    _validate_positive(distance_m, "distance_m")
    return float(distance_m / (charge_weight_kg ** (1.0 / 3.0)))


def estimate_ppv_mm_s(
    charge_weight_kg: float,
    distance_m: float,
    soil_type: str = "rock",
    site_constant: float = 1140.0,
    attenuation_exponent: float = -1.6,
) -> float:
    scaled_distance = scaled_distance_square_root(charge_weight_kg, distance_m)
    soil_factor = SOIL_ATTENUATION.get(str(soil_type).lower(), SOIL_ATTENUATION["rock"])
    return float(site_constant * (scaled_distance**attenuation_exponent) * soil_factor)


def ppv_frequency_product(ppv_mm_s: float, frequency_hz: float) -> float:
    return float(ppv_mm_s * frequency_hz)


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
