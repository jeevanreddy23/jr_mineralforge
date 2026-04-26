"""Geotechnical blast and seismic engineering calculations."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class BlastContext:
    charge_mass_kg: float
    distance_m: float
    soil_type: str = "rock"
    structure_distance_m: float | None = None


SOIL_ATTENUATION = {
    "rock": 1.00,
    "competent_rock": 0.92,
    "fractured_rock": 1.12,
    "sand": 1.25,
    "clay": 1.35,
    "fill": 1.45,
}


def scaled_distance_square_root(charge_mass_kg: float, distance_m: float) -> float:
    """Return square-root scaled distance, SD = D / sqrt(W)."""

    _validate_positive(charge_mass_kg, "charge_mass_kg")
    _validate_positive(distance_m, "distance_m")
    return float(distance_m / sqrt(charge_mass_kg))


def scaled_distance_cube_root(charge_mass_kg: float, distance_m: float) -> float:
    """Return cube-root scaled distance, SD = D / W^(1/3)."""

    _validate_positive(charge_mass_kg, "charge_mass_kg")
    _validate_positive(distance_m, "distance_m")
    return float(distance_m / (charge_mass_kg ** (1.0 / 3.0)))


def estimate_ppv_mm_s(
    charge_mass_kg: float,
    distance_m: float,
    site_constant: float = 1140.0,
    attenuation_exponent: float = -1.6,
    soil_type: str = "rock",
) -> float:
    """Estimate PPV from square-root scaled distance.

    The form PPV = k * SD^b is a common empirical blast-vibration relation.
    Site constants should be calibrated with local monitoring data before use
    in production.
    """

    sd = scaled_distance_square_root(charge_mass_kg, distance_m)
    soil_factor = SOIL_ATTENUATION.get(soil_type.lower(), SOIL_ATTENUATION["rock"])
    return float(site_constant * (sd**attenuation_exponent) * soil_factor)


def blast_feature_dict(context: BlastContext) -> dict[str, float]:
    structure_distance = context.structure_distance_m or context.distance_m
    return {
        "charge_mass_kg": float(context.charge_mass_kg),
        "distance_m": float(context.distance_m),
        "scaled_distance_sqrt": scaled_distance_square_root(context.charge_mass_kg, context.distance_m),
        "scaled_distance_cuberoot": scaled_distance_cube_root(context.charge_mass_kg, context.distance_m),
        "estimated_ppv_mm_s": estimate_ppv_mm_s(
            context.charge_mass_kg,
            context.distance_m,
            soil_type=context.soil_type,
        ),
        "structure_distance_m": float(structure_distance),
    }


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
