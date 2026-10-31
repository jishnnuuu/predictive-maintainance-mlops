import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models/xgboost_baseline.json"
SCHEMA_PATH = ROOT / "src/serving/feature_schema.json"

with SCHEMA_PATH.open(encoding="utf-8") as file:
    schema = json.load(file)

FEATURE_COLUMNS = schema["feature_columns"]
SAMPLE = schema["sample"]

model = xgb.XGBClassifier()
model.load_model(str(MODEL_PATH))

app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts machine failure risk using engineered telemetry features.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    features: dict[str, float]


@app.get("/")
def root():
    return {"message": "Predictive Maintenance API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": "xgboost_features"}


@app.get("/sample")
def sample():
    return {"features": SAMPLE}


@app.post("/predict")
def predict(request: PredictionRequest):
    received = set(request.features)
    expected = set(FEATURE_COLUMNS)

    missing = sorted(expected - received)
    extra = sorted(received - expected)

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "extra_features": extra},
        )

    try:
        row = pd.DataFrame(
            [[request.features[column] for column in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS,
        )

        probability = float(model.predict_proba(row)[0, 1])
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    threshold = 0.85

    return {
        "failure_probability": probability,
        "predicted_failure": probability >= threshold,
        "threshold": threshold,
    }