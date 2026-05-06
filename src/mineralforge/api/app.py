"""FastAPI app for blast vibration risk inference."""

from __future__ import annotations

from typing import Literal

from mineralforge.models.prediction import predict_record

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    FastAPI = None
    BaseModel = object


class BlastVibrationPayload(BaseModel):
    Charge_Weight_kg: float
    Burden_m: float
    Spacing_m: float
    Delay_ms: float = 0
    Soil_Type: str = "Medium"
    PPV_mm_s: float
    Frequency_Hz: float

    def to_model_record(self) -> dict:
        return {
            "Charge_Weight(kg)": self.Charge_Weight_kg,
            "Burden(m)": self.Burden_m,
            "Spacing(m)": self.Spacing_m,
            "Delay(ms)": self.Delay_ms,
            "Soil_Type": self.Soil_Type,
            "PPV(mm/s)": self.PPV_mm_s,
            "Frequency(Hz)": self.Frequency_Hz,
        }


if FastAPI is not None:
    app = FastAPI(title="MineralForge Blast Vibration Risk API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, Literal["ok"]]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(payload: BlastVibrationPayload) -> dict:
        return predict_record(payload.to_model_record())
else:
    app = None
