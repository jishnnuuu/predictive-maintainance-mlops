import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/features/baseline_features.csv"
OUTPUT_PATH = ROOT / "src/serving/feature_schema.json"

df = pd.read_csv(DATA_PATH, nrows=1)

excluded = {"datetime", "machineID", "failure_next_24h"}
feature_columns = [column for column in df.columns if column not in excluded]

schema = {
    "feature_columns": feature_columns,
    "sample": {
        column: float(df.iloc[0][column])
        for column in feature_columns
    },
}

OUTPUT_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")

print(f"Exported {len(feature_columns)} features to {OUTPUT_PATH}")