
import json
import time
from pathlib import Path

import pandas as pd
import xgboost as xgb

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


# ==================================================
# 1. PATHS AND MODEL LOADING
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models/xgboost_baseline.json"
SCHEMA_PATH = ROOT / "src/serving/feature_schema.json"

with SCHEMA_PATH.open(encoding="utf-8") as file:
    schema = json.load(file)

FEATURE_COLUMNS = schema["feature_columns"]
SAMPLE = schema["sample"]
THRESHOLD = schema.get("threshold", 0.85)

model = xgb.XGBClassifier()
model.load_model(str(MODEL_PATH))


# ==================================================
# 2. FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Predicts machine failure risk using "
        "engineered telemetry features."
    ),
    version="1.0.0",
)


# ==================================================
# 3. PROMETHEUS METRICS
# ==================================================

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
)

PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total model predictions",
    ["result"],
)

REQUEST_ERRORS = Counter(
    "api_errors_total",
    "Total failed API requests",
)

PREDICTION_PROBABILITY = Histogram(
    "model_prediction_probability",
    "Distribution of predicted failure probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)

FAILURE_RATE_GAUGE = Gauge(
    "model_failure_rate_current",
    "Rolling failure prediction rate (last 100 predictions)",
)

_recent_predictions: list = []


# ==================================================
# 4. REQUEST MONITORING MIDDLEWARE
# ==================================================

@app.middleware("http")
async def monitor_requests(request: Request, call_next):

    start_time = time.perf_counter()
    response = None

    try:
        response = await call_next(request)
        return response

    except Exception:
        raise

    finally:
        endpoint = request.url.path

        # Exclude Prometheus scrapes from application metrics.
        if endpoint != "/metrics":

            status = (
                str(response.status_code)
                if response is not None
                else "500"
            )

            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status=status,
            ).inc()

            REQUEST_LATENCY.labels(
                endpoint=endpoint,
            ).observe(
                time.perf_counter() - start_time
            )

            if response is None or response.status_code >= 400:
                REQUEST_ERRORS.inc()


# ==================================================
# 5. REQUEST SCHEMA
# ==================================================

class PredictionRequest(BaseModel):
    features: dict[str, float]


# ==================================================
# 6. API ENDPOINTS
# ==================================================

@app.get("/")
def root():
    return {
        "message": "Predictive Maintenance API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "xgboost_baseline",
        "features": len(FEATURE_COLUMNS),
    }


@app.get("/sample")
def sample():
    return {
        "features": SAMPLE
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    received = set(request.features)
    expected = set(FEATURE_COLUMNS)

    # Validate feature names.
    missing = sorted(expected - received)
    extra = sorted(received - expected)

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing,
                "extra_features": extra,
            },
        )

    try:
        # Preserve the feature order used during training.
        row = pd.DataFrame(
            [
                [
                    request.features[column]
                    for column in FEATURE_COLUMNS
                ]
            ],
            columns=FEATURE_COLUMNS,
        )

        probability = float(
            model.predict_proba(row)[0, 1]
        )

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    predicted_failure = probability >= THRESHOLD

    # Record prediction outcome.
    PREDICTION_COUNT.labels(
        result=(
            "failure"
            if predicted_failure
            else "no_failure"
        )
    ).inc()

    # Record probability distribution.
    PREDICTION_PROBABILITY.observe(probability)

    # Update rolling failure rate gauge (last 100 predictions).
    _recent_predictions.append(int(predicted_failure))
    if len(_recent_predictions) > 100:
        _recent_predictions.pop(0)
    FAILURE_RATE_GAUGE.set(
        sum(_recent_predictions) / len(_recent_predictions)
    )

    return {
        "failure_probability": probability,
        "predicted_failure": predicted_failure,
        "threshold": THRESHOLD,
    }


# ==================================================
# 7. PROMETHEUS METRICS ENDPOINT
# ==================================================

@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
