#!/usr/bin/env python3
"""
Test data generator for the Predictive Maintenance API.

Sends two waves of realistic predictions:

  Scenario 1 — Normal operation (Machine #1, healthy)
    500 requests with low-probability features → should produce
    near-zero failure probability and fill the "no_failure" counter.

  Scenario 2 — Pre-failure condition (Machine #2, degrading)
    200 requests with high-stress features → should produce high
    failure probability and trigger the "failure" counter + raise the
    failure_rate gauge on Grafana.

Usage:
    python tests/generate_test_data.py
    python tests/generate_test_data.py --url http://127.0.0.1:8000 --requests 1000
"""

import argparse
import json
import random
import sys
import time
import urllib.request
import urllib.error

API_URL = "http://127.0.0.1:8000"

# ─── Scenario 1: Healthy machine ────────────────────────────────────────────
HEALTHY_FEATURES = {
    "volt":      175.0,
    "rotate":    490.0,
    "pressure":  102.0,
    "vibration":  33.0,
    "age":        8.0,
    # 6-hour rolling stats — stable
    "volt_mean_6h":         175.5,  "volt_std_6h":         4.2,
    "rotate_mean_6h":       492.0,  "rotate_std_6h":       12.0,
    "pressure_mean_6h":     101.5,  "pressure_std_6h":      3.8,
    "vibration_mean_6h":     33.2,  "vibration_std_6h":     1.5,
    # 12-hour
    "volt_mean_12h":        175.0,  "volt_std_12h":         5.1,
    "rotate_mean_12h":      491.0,  "rotate_std_12h":      14.0,
    "pressure_mean_12h":    101.0,  "pressure_std_12h":     4.2,
    "vibration_mean_12h":    33.5,  "vibration_std_12h":    1.8,
    # 24-hour
    "volt_mean_24h":        174.5,  "volt_std_24h":         5.8,
    "rotate_mean_24h":      490.0,  "rotate_std_24h":      15.5,
    "pressure_mean_24h":    100.5,  "pressure_std_24h":     5.0,
    "vibration_mean_24h":    33.0,  "vibration_std_24h":    2.0,
    # 48-hour
    "volt_mean_48h":        174.0,  "volt_std_48h":         6.0,
    "rotate_mean_48h":      489.0,  "rotate_std_48h":      16.0,
    "pressure_mean_48h":    100.0,  "pressure_std_48h":     5.5,
    "vibration_mean_48h":    33.0,  "vibration_std_48h":    2.2,
    # Error counts — none
    "error1_count_24h":  0.0,
    "error2_count_24h":  0.0,
    "error3_count_24h":  0.0,
    "error4_count_24h":  0.0,
    "error5_count_24h":  0.0,
    "total_errors_24h":  0.0,
    # Time since maintenance — recent service
    "hours_since_last_maintenance": 120.0,
    # Machine model
    "model_model1": 1.0, "model_model2": 0.0,
    "model_model3": 0.0, "model_model4": 0.0,
}

# ─── Scenario 2: Degrading machine approaching failure ───────────────────────
FAILURE_FEATURES = {
    "volt":      155.0,   # low voltage — stress indicator
    "rotate":    560.0,   # over-spinning
    "pressure":  130.0,   # high pressure
    "vibration":  62.0,   # excessive vibration
    "age":        18.0,   # older machine
    # 6-hour: elevated and noisy
    "volt_mean_6h":         158.0,  "volt_std_6h":         18.0,
    "rotate_mean_6h":       555.0,  "rotate_std_6h":       45.0,
    "pressure_mean_6h":     128.0,  "pressure_std_6h":     22.0,
    "vibration_mean_6h":     60.0,  "vibration_std_6h":    12.0,
    # 12-hour
    "volt_mean_12h":        161.0,  "volt_std_12h":         20.0,
    "rotate_mean_12h":      550.0,  "rotate_std_12h":       50.0,
    "pressure_mean_12h":    125.0,  "pressure_std_12h":     25.0,
    "vibration_mean_12h":    58.0,  "vibration_std_12h":    14.0,
    # 24-hour
    "volt_mean_24h":        163.0,  "volt_std_24h":         22.0,
    "rotate_mean_24h":      545.0,  "rotate_std_24h":       55.0,
    "pressure_mean_24h":    122.0,  "pressure_std_24h":     28.0,
    "vibration_mean_24h":    56.0,  "vibration_std_24h":    15.0,
    # 48-hour — shows progressive degradation
    "volt_mean_48h":        165.0,  "volt_std_48h":         19.0,
    "rotate_mean_48h":      540.0,  "rotate_std_48h":       52.0,
    "pressure_mean_48h":    120.0,  "pressure_std_48h":     26.0,
    "vibration_mean_48h":    54.0,  "vibration_std_48h":    13.0,
    # Multiple recent errors
    "error1_count_24h":  3.0,
    "error2_count_24h":  2.0,
    "error3_count_24h":  1.0,
    "error4_count_24h":  2.0,
    "error5_count_24h":  1.0,
    "total_errors_24h":  9.0,
    # Long time since last maintenance
    "hours_since_last_maintenance": 720.0,
    # Older model
    "model_model1": 0.0, "model_model2": 0.0,
    "model_model3": 0.0, "model_model4": 1.0,
}


def jitter(features: dict, noise: float = 0.03) -> dict:
    """Add small random noise so each request is unique."""
    return {
        k: v * (1 + random.uniform(-noise, noise)) if isinstance(v, float) else v
        for k, v in features.items()
    }


def predict(base_features: dict, url: str) -> dict | None:
    payload = json.dumps({"features": jitter(base_features)}).encode()
    req = urllib.request.Request(
        f"{url}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return None


def run_scenario(
    label: str,
    features: dict,
    n: int,
    url: str,
    delay: float = 0.05,
):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Sending {n} requests to {url}/predict  (delay={delay}s)")
    print(f"{'='*60}")

    failures = normal = errors = 0
    for i in range(1, n + 1):
        result = predict(features, url)
        if result is None:
            errors += 1
        elif result.get("predicted_failure"):
            failures += 1
        else:
            normal += 1

        if i % 50 == 0 or i == n:
            prob = result.get("failure_probability", "?") if result else "?"
            print(
                f"  [{i:4d}/{n}]  "
                f"last_prob={prob:.4f}  "
                f"failures={failures}  normal={normal}  errors={errors}"
                if isinstance(prob, float)
                else f"  [{i:4d}/{n}]  errors={errors}"
            )

        time.sleep(delay)

    print(f"\n  ✅ Done — failures={failures}, normal={normal}, errors={errors}")
    return failures, normal, errors


def wait_for_api(url: str, retries: int = 10):
    print(f"Waiting for API at {url}/health …")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=3) as r:
                body = json.loads(r.read())
                if body.get("status") == "healthy":
                    print(f"  API is healthy — {body}")
                    return True
        except Exception:
            pass
        time.sleep(2)
        print(f"  Retrying ({attempt + 1}/{retries}) …")
    print("  ❌ API did not become healthy", file=sys.stderr)
    return False


def main():
    parser = argparse.ArgumentParser(description="Generate test data for the API")
    parser.add_argument("--url", default=API_URL)
    parser.add_argument("--requests", type=int, default=500,
                        help="Number of healthy-scenario requests")
    parser.add_argument("--failure-requests", type=int, default=200,
                        help="Number of failure-scenario requests")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Seconds between requests")
    args = parser.parse_args()

    if not wait_for_api(args.url):
        sys.exit(1)

    # ── Scenario 1: Normal operation ────────────────────────────────────────
    run_scenario(
        label="SCENARIO 1 — Normal Operation (healthy machine)",
        features=HEALTHY_FEATURES,
        n=args.requests,
        url=args.url,
        delay=args.delay,
    )

    print("\n⏸  Pausing 3 seconds before failure scenario …")
    time.sleep(3)

    # ── Scenario 2: Impending failure ───────────────────────────────────────
    run_scenario(
        label="SCENARIO 2 — Pre-Failure Condition (degrading machine)",
        features=FAILURE_FEATURES,
        n=args.failure_requests,
        url=args.url,
        delay=args.delay,
    )

    print("\n🎉  Test data generation complete.")
    print("    Open Grafana at http://127.0.0.1:3000 to see the panels update.")


if __name__ == "__main__":
    main()
