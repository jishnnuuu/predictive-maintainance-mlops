# Predictive Maintenance MLOps

An end-to-end MLOps pipeline that predicts machine failures 24 hours in advance using the **Microsoft Azure Predictive Maintenance** dataset. Demonstrates every layer of a production-grade ML system: data versioning, experiment tracking, containerisation, orchestration, and monitoring.

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Dataset](#2-dataset)
3. [Architecture Overview](#3-architecture-overview)
4. [MLOps Stack](#4-mlops-stack)
5. [Repository Layout](#5-repository-layout)
6. [Quick Start — Local Setup](#6-quick-start--local-setup)
7. [Reproducing the Full Pipeline](#7-reproducing-the-full-pipeline)
8. [Running the API](#8-running-the-api)
9. [Monitoring with Prometheus & Grafana](#9-monitoring-with-prometheus--grafana)
10. [Experiment Tracking with MLflow](#10-experiment-tracking-with-mlflow)
11. [Kubernetes Deployment](#11-kubernetes-deployment)
12. [CI/CD with GitHub Actions](#12-cicd-with-github-actions)
13. [Running Tests](#13-running-tests)
14. [Model Results](#14-model-results)
15. [Pipeline Parameters](#15-pipeline-parameters)

---

## 1. Project Goal

> **For every machine and every hour, predict whether that machine will experience a failure within the next 24 hours.**

Early warning enables maintenance teams to act before a breakdown occurs, reducing downtime and repair cost.

---

## 2. Dataset

**Source:** [Microsoft Azure Predictive Maintenance — Kaggle](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)

Download the dataset and place all five CSV files in `data/raw/`:

```
data/raw/
├── PdM_telemetry.csv   # hourly sensor readings (volt, rotate, pressure, vibration)
├── PdM_errors.csv      # 5 error types logged at event time
├── PdM_maint.csv       # component replacement records
├── PdM_failures.csv    # actual failure events (ground truth)
└── PdM_machines.csv    # machine metadata (model type, age)
```

| Attribute | Value |
|---|---|
| Machines | 100 |
| Time span | Jan 2015 – Jan 2016 (hourly) |
| Raw rows | 876,100 |
| Feature rows (after engineering) | 871,400 |
| Positive label rate | ~1.96% (class-imbalanced) |

---

## 3. Architecture Overview

```
Raw CSVs  ──►  DVC prepare  ──►  DVC features  ──►  DVC train  ──►  DVC evaluate
                                                         │
                                                    ┌────▼────┐
                                                    │ MLflow  │
                                                    │Tracking │
                                                    └────┬────┘
                                                         │
                                                    ┌────▼────┐      ┌─────────────┐
                                                    │ FastAPI │◄─────│ Docker      │
                                                    │   API   │      │ Compose /   │
                                                    └────┬────┘      │ Kubernetes  │
                                                         │           └─────────────┘
                                              ┌──────────▼──────────┐
                                              │  /metrics endpoint  │
                                              └──────────┬──────────┘
                                                         │
                                              ┌──────────▼──────────┐
                                              │    Prometheus        │
                                              └──────────┬──────────┘
                                                         │
                                              ┌──────────▼──────────┐
                                              │      Grafana         │
                                              └─────────────────────┘
```

---

## 4. MLOps Stack

| Tool | Role |
|---|---|
| **DVC** | Pipeline reproducibility, data & model versioning |
| **MLflow** | Experiment tracking — params, metrics, artifacts for all 3 models |
| **Docker + Compose** | Containerise API + Prometheus + Grafana as a single stack |
| **FastAPI** | Async prediction REST API with Pydantic v2 validation |
| **Prometheus** | Pull-based metrics scraping from `/metrics` every 5 s |
| **Grafana** | Pre-provisioned dashboard wired to Prometheus |
| **GitHub Actions** | Lint → test → DVC pipeline → Docker build on every push |
| **Kubernetes** | Deployment, Service, and HPA manifests for production |
| **pytest + ruff** | Unit tests and linting |

---

## 5. Repository Layout

```
.
├── .github/workflows/pipeline.yml   # CI/CD pipeline
├── data/
│   ├── raw/                         # Raw CSVs (git-ignored, DVC-tracked)
│   ├── processed/                   # machine_hourly.csv (DVC output)
│   └── features/                    # baseline_features.csv (DVC output)
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── metrics/                         # JSON metrics files (DVC outputs)
├── models/                          # Trained model files (DVC outputs)
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboards/
│       └── provisioning/
├── src/
│   ├── data/
│   │   └── prepare.py               # DVC stage: prepare
│   ├── features/
│   │   └── build_features.py        # DVC stage: features
│   ├── models/
│   │   ├── train.py                 # DVC stage: train (XGBoost)
│   │   ├── train_lstm.py            # DVC stage: train_lstm
│   │   ├── train_transformer.py     # DVC stage: train_transformer
│   │   └── evaluate.py              # DVC stage: evaluate
│   └── serving/
│       ├── app.py                   # FastAPI application
│       ├── feature_schema.json      # Feature list + serving threshold
│       └── export_schema.py
├── tests/
│   ├── test_pipeline.py             # Unit tests
│   └── create_ci_dataset.py         # Synthetic data for CI
├── docker-compose.yml
├── Dockerfile
├── dvc.yaml                         # DVC pipeline definition
├── dvc.lock                         # DVC pipeline lock file
├── params.yaml                      # All configurable parameters
├── requirements.txt
├── requirements-dev.txt
└── requirements-serving.txt
```

---

## 6. Quick Start — Local Setup

### Prerequisites

- Python 3.11
- Git
- Docker + Docker Compose
- `brew install libomp` (macOS only — required by XGBoost)

### Step 1 — Clone and create virtual environment

```bash
git clone <repo-url>
cd predictive-maintainance-mlops

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### Step 2 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3 — Add the raw dataset

Download from [Kaggle](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) and place the five CSV files in `data/raw/`:

```bash
mkdir -p data/raw
# Copy PdM_telemetry.csv, PdM_errors.csv, PdM_maint.csv,
#         PdM_failures.csv, PdM_machines.csv  →  data/raw/
```

---

## 7. Reproducing the Full Pipeline

Run all stages in order, or use `dvc repro` once a DVC remote is configured.

### Option A — Run each stage manually

```bash
# Stage 1: prepare — merge tables, create 24h failure labels
python -m src.data.prepare

# Stage 2: features — rolling stats, error counts, maintenance features
python -m src.features.build_features

# Stage 3: train XGBoost + log to MLflow
python -m src.models.train

# Stage 4: evaluate — write metrics/baseline_metrics.json
python -m src.models.evaluate

# Stage 5: train LSTM  (~30 min on CPU)
python -m src.models.train_lstm

# Stage 6: train Transformer  (~30 min on CPU)
python -m src.models.train_transformer
```

### Option B — DVC repro (after configuring a remote)

```bash
# Configure a local DVC remote (example — use S3/GCS in production)
dvc remote add -d myremote /tmp/dvc-remote
dvc push                    # push data/models to remote

# Reproduce the full pipeline (only re-runs changed stages)
dvc repro

# Compare metrics across runs
dvc metrics show
dvc metrics diff
```

### All configurable parameters live in `params.yaml`

```yaml
data:
  prediction_horizon_hours: 24   # label window

split:
  train_end:       "2015-08-31 23:00:00"
  validation_end:  "2015-10-31 23:00:00"
  test_end:        "2015-12-31 23:00:00"

models:
  baseline:
    n_estimators: 300
    max_depth:    6
    learning_rate: 0.05
  sequence:
    window_hours: 24
    hidden_size:  64
    num_layers:   2
  transformer:
    window_hours: 24
    d_model: 64
    nhead:   4
    num_layers: 2

training:
  batch_size: 256
  epochs:     10
  learning_rate: 0.0001
  early_stopping_patience: 3
```

---

## 8. Running the API

### Locally (no Docker)

```bash
source .venv/bin/activate
uvicorn src.serving.app:app --host 127.0.0.1 --port 8000 --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive Swagger UI.

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check — model name + feature count |
| `GET` | `/sample` | Example request payload |
| `POST` | `/predict` | Predict failure probability |
| `GET` | `/metrics` | Prometheus metrics |

### Example prediction request

```bash
# Get a valid sample payload
curl http://127.0.0.1:8000/sample

# Send a prediction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "volt": 174.1, "rotate": 489.2, "pressure": 91.9, "vibration": 32.3,
      "age": 18.0,
      "volt_mean_6h": 178.4, "volt_std_6h": 8.1,
      ...
    }
  }'

# Response
{
  "failure_probability": 0.000003,
  "predicted_failure": false,
  "threshold": 0.89
}
```

---

## 9. Monitoring with Prometheus & Grafana

### Start the full observability stack

```bash
docker compose up -d
```

This starts three containers:

| Service | URL | Credentials |
|---|---|---|
| FastAPI API | http://127.0.0.1:8000 | — |
| Prometheus | http://127.0.0.1:9090 | — |
| Grafana | http://127.0.0.1:3000 | admin / admin |

### Grafana dashboard

A pre-provisioned dashboard (**Predictive Maintenance Monitoring**) is automatically loaded. It shows:

- API requests per second
- Total predictions (failure vs no-failure)
- Prediction distribution (pie chart)
- Average API latency (histogram)
- API error count

### Stop the stack

```bash
docker compose down
```

### Prometheus metrics exposed

| Metric | Type | Description |
|---|---|---|
| `api_requests_total` | Counter | Requests by method, endpoint, status |
| `api_request_latency_seconds` | Histogram | Request latency by endpoint |
| `model_predictions_total` | Counter | Predictions by result (failure / no_failure) |
| `api_errors_total` | Counter | Total failed requests |

---

## 10. Experiment Tracking with MLflow

All three models log to the same `predictive-maintenance` experiment.

```bash
# Launch the MLflow UI
mlflow ui --host 127.0.0.1 --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

Each run logs:

- **Params:** model type, hyperparameters, class weights
- **Metrics:** PR-AUC, Precision, Recall, F1 (validation + test)
- **Artifacts:** model file, feature names, metrics JSON
- **Training history:** per-epoch loss and val PR-AUC (LSTM/Transformer)

### Compare runs from the CLI

```bash
mlflow experiments search --experiment-name predictive-maintenance
```

---

## 11. Kubernetes Deployment

### Prerequisites

- kubectl
- A local cluster: [minikube](https://minikube.sigs.k8s.io/) or [kind](https://kind.sigs.k8s.io/)

### Build and load the image

```bash
# Build
docker build -t predictive-maintenance-api:latest .

# Load into minikube
minikube image load predictive-maintenance-api:latest
```

### Deploy

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Verify
kubectl get pods
kubectl get svc predictive-maintenance-api
```

### Access the API

```bash
# minikube
minikube service predictive-maintenance-api --url

# kind / bare cluster
kubectl port-forward svc/predictive-maintenance-api 8000:80
# then open http://127.0.0.1:8000
```

### HPA — Horizontal Pod Autoscaler

The HPA scales the deployment between **2 and 6 replicas** based on:
- CPU utilisation > 70%
- Memory utilisation > 80%

---

## 12. CI/CD with GitHub Actions

The workflow file at [`.github/workflows/pipeline.yml`](.github/workflows/pipeline.yml) runs on every push and pull request to `main`.

### Pipeline stages

```
push to main
     │
     ▼
┌─────────┐    ┌──────────┐    ┌─────────────────┐    ┌──────────────┐
│  lint   │───►│  tests   │───►│  DVC pipeline   │───►│ Docker build │
│ (ruff)  │    │ (pytest) │    │  (XGBoost only) │    │ + smoke test │
└─────────┘    └──────────┘    └─────────────────┘    └──────────────┘
```

1. **Lint** — `ruff check src/`
2. **Test** — `pytest tests/ -v`
3. **Train** — runs prepare → features → train → evaluate with a synthetic CI dataset
4. **Docker** — builds the image and runs a live `/health` smoke test

> **Note:** LSTM and Transformer are excluded from CI to keep the job under 10 minutes. Run them locally for full training.

---

## 13. Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

The test suite covers:

- Label creation correctness (no leakage, correct 24h window)
- Feature engineering (rolling stats are strictly backward-looking)
- Threshold search (valid range, returns float)
- API endpoints (`/health`, `/sample`, `/predict`, `/metrics`)
- Missing-feature validation (422 on incomplete request)

---

## 14. Model Results

All results are on the **held-out test set** (Nov–Dec 2015).

| Model | PR-AUC | Precision | Recall | F1 | Threshold |
|---|---|---|---|---|---|
| **XGBoost** ✅ serving | 0.9998 | 0.9981 | 0.9966 | 0.9974 | 0.89 |
| LSTM (10 epochs) | 0.8129 | 0.6774 | 0.8188 | 0.7414 | 0.98 |
| Transformer (10 epochs) | 0.9768 | 0.9242 | 0.9369 | 0.9305 | 0.96 |

> XGBoost dominates on this tabular dataset. Deep models benefit from their ability to model temporal dynamics but require full training to reach competitive performance.

The optimal threshold is automatically determined on the validation set and written to `src/serving/feature_schema.json` — the API reads it at startup, so no manual updates are needed after retraining.

---

## 15. Pipeline Parameters

All parameters are in [`params.yaml`](params.yaml). Changing any value and running `dvc repro` (or the relevant stage manually) will re-run only the affected downstream stages.

| Parameter | Default | Effect |
|---|---|---|
| `data.prediction_horizon_hours` | 24 | Redefines the label window |
| `data.history_hours` | 48 | Minimum observations needed per machine |
| `features.rolling_windows` | [6, 12, 24, 48] | Rolling window sizes for telemetry stats |
| `features.error_window_hours` | 24 | Error count lookback |
| `models.baseline.*` | XGBoost hyperparams | Affects train + evaluate stages |
| `models.sequence.*` | LSTM architecture | Affects train_lstm stage |
| `models.transformer.*` | Transformer architecture | Affects train_transformer stage |
| `training.epochs` | 10 | Deep model training budget |
| `training.early_stopping_patience` | 3 | Stop if val PR-AUC doesn't improve |
| `split.*` | 2015 date boundaries | Affects all train + evaluate stages |



