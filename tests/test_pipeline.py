"""
Unit tests for the Predictive Maintenance MLOps project.
Run with:  pytest tests/ -v
"""

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

class TestCreateFailureLabel:
    """Tests for src.data.prepare.create_failure_label"""

    def _make_telemetry(self):
        return pd.DataFrame({
            "machineID": [1, 1, 1, 1, 1],
            "datetime": pd.to_datetime([
                "2015-01-01 00:00:00",
                "2015-01-01 01:00:00",
                "2015-01-01 02:00:00",
                "2015-01-01 22:00:00",
                "2015-01-02 00:00:00",
            ]),
            "volt": [170.0] * 5,
            "rotate": [450.0] * 5,
            "pressure": [100.0] * 5,
            "vibration": [40.0] * 5,
        })

    def _make_failures(self):
        return pd.DataFrame({
            "machineID": [1],
            "datetime": pd.to_datetime(["2015-01-01 23:00:00"]),
            "failure": ["comp1"],
        })

    def test_positive_label_within_24h(self):
        from src.data.prepare import create_failure_label
        tel = self._make_telemetry()
        fail = self._make_failures()
        result = create_failure_label(tel, fail)

        # Row at 01:00 is 22 hours before failure — should be positive
        row = result[result["datetime"] == pd.Timestamp("2015-01-01 01:00:00")]
        assert row["failure_next_24h"].values[0] == 1

    def test_negative_label_after_failure(self):
        from src.data.prepare import create_failure_label
        tel = self._make_telemetry()
        fail = self._make_failures()
        result = create_failure_label(tel, fail)

        # Row at 00:00 the day after failure — should be negative
        row = result[result["datetime"] == pd.Timestamp("2015-01-02 00:00:00")]
        assert row["failure_next_24h"].values[0] == 0

    def test_no_future_leakage(self):
        """failure_datetime must not appear in output columns."""
        from src.data.prepare import create_failure_label
        tel = self._make_telemetry()
        fail = self._make_failures()
        result = create_failure_label(tel, fail)
        assert "failure_datetime" not in result.columns

    def test_label_dtype(self):
        from src.data.prepare import create_failure_label
        tel = self._make_telemetry()
        fail = self._make_failures()
        result = create_failure_label(tel, fail)
        assert result["failure_next_24h"].dtype in (np.dtype("int8"), np.dtype("int64"))


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

class TestTelemetryFeatures:
    """Tests for src.features.build_features.add_telemetry_features"""

    def _make_df(self, n=50):
        rng = np.random.default_rng(0)
        times = pd.date_range("2015-01-01", periods=n, freq="h")
        return pd.DataFrame({
            "machineID": [1] * n,
            "datetime": times,
            "volt": rng.normal(170, 10, n),
            "rotate": rng.normal(450, 40, n),
            "pressure": rng.normal(100, 8, n),
            "vibration": rng.normal(40, 5, n),
        })

    def test_rolling_columns_created(self):
        from src.features.build_features import add_telemetry_features
        df = self._make_df()
        result = add_telemetry_features(df, rolling_windows=[6])
        assert "volt_mean_6h" in result.columns
        assert "volt_std_6h" in result.columns

    def test_no_future_leakage_in_rolling(self):
        """Rolling stats at position i must not use row i+1 onward."""
        from src.features.build_features import add_telemetry_features
        df = self._make_df(n=10)
        result = add_telemetry_features(df, rolling_windows=[3])
        # The first two rows have fewer than 3 observations — must be NaN
        assert pd.isna(result["volt_mean_3h"].iloc[0])
        assert pd.isna(result["volt_mean_3h"].iloc[1])
        # Row index 2 should have a valid value (3 observations available)
        assert not pd.isna(result["volt_mean_3h"].iloc[2])


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class TestThresholdSearch:
    """Tests for find_best_threshold in train.py"""

    def test_returns_float(self):
        from src.models.train import find_best_threshold
        y = np.array([0, 0, 1, 1, 0, 1])
        probs = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        t = find_best_threshold(y, probs)
        assert isinstance(t, float)

    def test_threshold_in_valid_range(self):
        from src.models.train import find_best_threshold
        y = np.array([0, 0, 1, 1, 0, 1])
        probs = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        t = find_best_threshold(y, probs)
        assert 0.0 < t < 1.0

    def test_metrics_dict_keys(self):
        from src.models.train import calculate_metrics
        y = np.array([0, 1, 1, 0])
        probs = np.array([0.2, 0.8, 0.7, 0.3])
        m = calculate_metrics(y, probs, threshold=0.5)
        assert set(m.keys()) == {"pr_auc", "precision", "recall", "f1", "threshold"}


# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

SCHEMA_PATH = Path("src/serving/feature_schema.json")


@pytest.fixture(scope="module")
def client():
    """Return a FastAPI test client with the XGBoost model mocked out.

    Module-scoped so the app (and its Prometheus metrics) is only
    instantiated once per test session — prometheus_client raises
    ValueError('Duplicated timeseries') if the same Counter is registered
    more than once in the same process.
    """
    if not SCHEMA_PATH.exists():
        pytest.skip("feature_schema.json not found — run the pipeline first")

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])

    import xgboost as xgb
    with patch.object(xgb, "XGBClassifier", return_value=mock_model):
        from src.serving.app import app
        return TestClient(app)


class TestAPIEndpoints:

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "message" in r.json()

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert "features" in body

    def test_sample_returns_features(self, client):
        r = client.get("/sample")
        assert r.status_code == 200
        assert "features" in r.json()

    def test_predict_valid_input(self, client):
        with open(SCHEMA_PATH, encoding="utf-8") as f:
            schema = json.load(f)
        r = client.post("/predict", json={"features": schema["sample"]})
        assert r.status_code == 200
        body = r.json()
        assert "failure_probability" in body
        assert "predicted_failure" in body
        assert "threshold" in body

    def test_predict_missing_features(self, client):
        r = client.post("/predict", json={"features": {"volt": 170.0}})
        assert r.status_code == 422

    def test_metrics_endpoint(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert b"api_requests_total" in r.content
