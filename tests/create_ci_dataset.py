"""
Creates a tiny synthetic dataset that mirrors the real Microsoft Azure PdM
CSV schema so the DVC pipeline stages can be validated in CI without the
full 76 MB telemetry file.
"""

import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

# ── Parameters ───────────────────────────────────────────────
N_MACHINES = 10
HOURS = 24 * 100  # 100 days of hourly readings
START = pd.Timestamp("2015-08-01 00:00:00")

# ── Telemetry ────────────────────────────────────────────────
rows = []
for machine_id in range(1, N_MACHINES + 1):
    times = pd.date_range(START, periods=HOURS, freq="h")
    for t in times:
        rows.append({
            "datetime": t,
            "machineID": machine_id,
            "volt":      rng.normal(170, 15),
            "rotate":    rng.normal(450, 50),
            "pressure":  rng.normal(100, 10),
            "vibration": rng.normal(40, 5),
        })
telemetry = pd.DataFrame(rows)
telemetry.to_csv(RAW / "PdM_telemetry.csv", index=False)
print(f"Telemetry: {len(telemetry):,} rows")

# ── Failures ─────────────────────────────────────────────────
failure_rows = []
for machine_id in range(1, N_MACHINES + 1):
    # One failure per machine in the last 15 days
    fail_time = START + pd.Timedelta(days=rng.integers(16, 29))
    failure_rows.append({
        "datetime":  fail_time,
        "machineID": machine_id,
        "failure":   f"comp{rng.integers(1, 5)}",
    })
failures = pd.DataFrame(failure_rows)
failures.to_csv(RAW / "PdM_failures.csv", index=False)
print(f"Failures: {len(failures)} rows")

# ── Errors ───────────────────────────────────────────────────
error_rows = []
for machine_id in range(1, N_MACHINES + 1):
    for _ in range(rng.integers(2, 10)):
        error_rows.append({
            "datetime":  START + pd.Timedelta(hours=int(rng.integers(0, HOURS))),
            "machineID": machine_id,
            "errorID":   f"error{rng.integers(1, 6)}",
        })
errors = pd.DataFrame(error_rows)
errors.to_csv(RAW / "PdM_errors.csv", index=False)
print(f"Errors: {len(errors)} rows")

# ── Maintenance ──────────────────────────────────────────────
maint_rows = []
for machine_id in range(1, N_MACHINES + 1):
    for _ in range(rng.integers(1, 4)):
        maint_rows.append({
            "datetime":  START + pd.Timedelta(hours=int(rng.integers(0, HOURS))),
            "machineID": machine_id,
            "comp":      f"comp{rng.integers(1, 5)}",
        })
maint = pd.DataFrame(maint_rows)
maint.to_csv(RAW / "PdM_maint.csv", index=False)
print(f"Maintenance: {len(maint)} rows")

# ── Machines ─────────────────────────────────────────────────
machines = pd.DataFrame({
    "machineID": range(1, N_MACHINES + 1),
    "model":     [f"model{((i - 1) % 4) + 1}" for i in range(1, N_MACHINES + 1)],
    "age":       rng.integers(1, 20, size=N_MACHINES),
})
machines.to_csv(RAW / "PdM_machines.csv", index=False)
print(f"Machines: {len(machines)} rows")

print("\nSynthetic dataset created in data/raw/")
