# Predictive Maintenance MLOps

Predictive maintenance system for machine telemetry data.

## Objective

For every machine and hour t, predict whether the machine will
experience a failure within the next 24 hours.

## Dataset

Microsoft Azure Predictive Maintenance dataset.

The dataset contains:

- telemetry
- errors
- maintenance records
- failures
- machine metadata

## ML Models

1. XGBoost baseline
2. GRU/LSTM
3. Transformer encoder

## MLOps Stack

- Git/GitHub
- DVC
- MLflow
- Docker
- GitHub Actions / Jenkins
- Kubernetes
- FastAPI
- Prometheus
- Grafana

## PE MidTerm

## Development

Create environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
