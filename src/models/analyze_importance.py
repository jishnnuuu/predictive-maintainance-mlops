from pathlib import Path

import json
import pandas as pd
import xgboost as xgb


FEATURE_PATH = Path(
    "data/features/baseline_features.csv"
)

MODEL_PATH = Path(
    "models/xgboost_baseline.json"
)

OUTPUT_PATH = Path(
    "metrics/feature_importance.csv"
)

TARGET = "failure_next_24h"

DROP_COLUMNS = [
    "datetime",
    "machineID",
    TARGET,
]


def main():
    print("Loading feature dataset...")

    df = pd.read_csv(
        FEATURE_PATH
    )

    X = df.drop(
        columns=DROP_COLUMNS
    )

    print(
        f"Number of features: {X.shape[1]}"
    )

    print("\nLoading XGBoost model...")

    model = xgb.XGBClassifier()

    model.load_model(
        MODEL_PATH
    )

    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------

    importance = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": importance,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # Print top features
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TOP 20 FEATURES")
    print("=" * 60)

    print(
        importance_df
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------
    # Feature groups
    # --------------------------------------------------

    def feature_group(feature):

        if feature.startswith(
            ("volt_", "rotate_", "pressure_", "vibration_")
        ):
            return "telemetry"

        if feature.startswith(
            ("error1_", "error2_", "error3_",
             "error4_", "error5_", "total_errors")
        ):
            return "errors"

        if feature == "hours_since_last_maintenance":
            return "maintenance"

        if feature.startswith("model_"):
            return "machine_model"

        return "other"

    importance_df["group"] = (
        importance_df["feature"]
        .apply(feature_group)
    )

    group_importance = (
        importance_df
        .groupby("group")["importance"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print("\n" + "=" * 60)
    print("FEATURE GROUP IMPORTANCE")
    print("=" * 60)

    print(
        group_importance
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    importance_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nFeature importance saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()