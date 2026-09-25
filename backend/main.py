from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from train_model import FEATURES, train

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
ANALYTICS_PATH = ROOT / "models" / "analytics.json"

app = FastAPI(title="HeartAI API", version="1.0.0", description="Educational heart disease risk classification API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class PatientFeatures(BaseModel):
    age: float = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: float = Field(ge=50, le=250)
    chol: float = Field(ge=50, le=700)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(ge=50, le=250)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=-5, le=10)
    slope: int = Field(ge=0, le=2)
    ca: float = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=3)


def ensure_model() -> dict[str, Any]:
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        try:
            train()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model is not trained. Run python train_model.py first. ({exc})") from exc
    return joblib.load(MODEL_PATH)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "HeartAI API"}

@app.get("/model-metrics")
def model_metrics() -> dict[str, Any]:
    ensure_model()
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

@app.get("/analytics")
def analytics() -> dict[str, Any]:
    ensure_model()
    rows = pd.read_json(ANALYTICS_PATH)
    return {
        "rows": len(rows),
        "target_distribution": rows["target"].value_counts().sort_index().to_dict(),
        "means": {column: round(float(rows[column].mean()), 2) for column in ["age", "chol", "trestbps", "thalach"]},
    }

@app.post("/predict")
def predict(patient: PatientFeatures) -> dict[str, Any]:
    bundle = ensure_model()
    values = patient.model_dump()
    frame = pd.DataFrame([values], columns=FEATURES)
    model = bundle["model"]
    probability = float(model.predict_proba(frame)[0][1])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "risk_probability": round(probability, 4),
        "message": "Higher risk detected" if prediction else "Lower risk detected",
        "model": bundle["model_name"],
        "disclaimer": "Educational estimate only; this is not a medical diagnosis.",
    }
