"""FastAPI wrapper for real-time edge inference."""

from __future__ import annotations

from typing import Literal

from mineralforge.pipeline import RockBurstPipeline

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except Exception:  # pragma: no cover - optional app dependency
    FastAPI = None
    BaseModel = object


if FastAPI is not None:
    app = FastAPI(title="MineralForge Risk API", version="0.2.0")
    pipeline = RockBurstPipeline.demo()
else:
    app = None
    pipeline = None


class FeaturePayload(BaseModel):
    zone: str = "Stope 3"
    acoustic_rms: float
    acoustic_peak_frequency_hz: float
    spectral_entropy: float
    acoustic_event_rate: float
    vibration_ppv: float
    vibration_dominant_frequency_hz: float
    vibration_energy: float
    cumulative_energy: float


if FastAPI is not None:

    @app.get("/health")
    def health() -> dict[str, Literal["ok"]]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(payload: FeaturePayload) -> dict:
        features = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        zone = str(features.pop("zone"))
        return pipeline.assess(features, zone=zone).to_dict()
