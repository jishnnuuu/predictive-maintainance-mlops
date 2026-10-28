from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


FEATURE_PATH = Path(
    "data/features/baseline_features.csv"
)

MODEL_PATH = Path(
    "models/xgboost_baseline.json"
)

TARGET = "failure_next_24h"

DROP_COLUMNS = [
    "datetime",
    "machineID",
    TARGET,
]


def load_data():

    df = pd.read_csv(
        FEATURE_PATH
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    return df.sort_values(
        ["datetime", "machineID"]
    ).reset_index(drop=True)


def split_data(df):

    test = df[
        df["datetime"] >= "2015-11-01"
    ].copy()

    return test


def prepare_xy(df):

    X = df.drop(
        columns=DROP_COLUMNS
    )

    y = df[TARGET]

    return X, y


def calculate_warning_hours(df):

    """
    For every positive prediction timestamp,
    calculate how many hours remain until
    the corresponding failure.

    The feature dataset only contains the binary
    failure_next_24h label, so we reconstruct the
    actual next failure timestamp from the raw
    failure dataset.
    """

    failures = pd.read_csv(
        "data/raw/PdM_failures.csv"
    )

    failures["datetime"] = pd.to_datetime(
        failures["datetime"]
    )

    failures = failures.sort_values(
        ["machineID", "datetime"]
    )

    failure_times = failures[
        ["machineID", "datetime"]
    ].rename(
        columns={
            "datetime": "failure_datetime"
        }
    )

    result = pd.merge_asof(
        df.sort_values(
            ["datetime", "machineID"]
        ),
        failure_times.sort_values(
            ["failure_datetime", "machineID"]
        ),
        left_on="datetime",
        right_on="failure_datetime",
        by="machineID",
        direction="forward",
        allow_exact_matches=False,
    )

    result["hours_to_failure"] = (
        result["failure_datetime"]
        - result["datetime"]
    ).dt.total_seconds() / 3600

    return result


def main():

    print("Loading feature dataset...")

    df = load_data()

    test = split_data(df)

    print(
        f"Test rows: {len(test):,}"
    )

    X_test, y_test = prepare_xy(test)

    print("\nLoading model...")

    model = xgb.XGBClassifier()

    model.load_model(
        MODEL_PATH
    )

    print("\nGenerating predictions...")

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test = test.copy()

    test["probability"] = probabilities

    # Use the threshold selected on validation.
    threshold = 0.95

    test["prediction"] = (
        test["probability"] >= threshold
    ).astype(int)

    print(
        "\nReconstructing failure timing..."
    )

    test = calculate_warning_hours(
        test
    )

    # --------------------------------------------------
    # Only evaluate actual positive cases
    # --------------------------------------------------

    positive_cases = test[
        test[TARGET] == 1
    ].copy()

    # --------------------------------------------------
    # Warning-time buckets
    # --------------------------------------------------

    positive_cases["warning_bucket"] = pd.cut(
        positive_cases["hours_to_failure"],
        bins=[
            0,
            1,
            6,
            12,
            18,
            24,
        ],
        labels=[
            "0-1h",
            "1-6h",
            "6-12h",
            "12-18h",
            "18-24h",
        ],
        include_lowest=True,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "POSITIVE CASES BY WARNING TIME"
    )

    print(
        "=" * 70
    )

    distribution = (
        positive_cases[
            "warning_bucket"
        ]
        .value_counts(
            sort=False
        )
    )

    percentages = (
        distribution
        / len(positive_cases)
        * 100
    )

    warning_summary = pd.DataFrame(
        {
            "cases": distribution,
            "percentage": percentages,
        }
    )

    print(
        warning_summary
    )

    # --------------------------------------------------
    # Model detection by warning bucket
    # --------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL DETECTION BY WARNING TIME"
    )

    print(
        "=" * 70
    )

    rows = []

    for bucket in [
        "0-1h",
        "1-6h",
        "6-12h",
        "12-18h",
        "18-24h",
    ]:

        subset = positive_cases[
            positive_cases[
                "warning_bucket"
            ] == bucket
        ]

        if len(subset) == 0:
            continue

        detected = (
            subset["prediction"] == 1
        ).sum()

        detection_rate = (
            detected / len(subset)
        )

        rows.append(
            {
                "warning_bucket": bucket,
                "cases": len(subset),
                "detected": detected,
                "detection_rate": detection_rate,
            }
        )

    detection_df = pd.DataFrame(
        rows
    )

    print(
        detection_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    output_path = Path(
        "metrics/warning_time_analysis.csv"
    )

    detection_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nResults saved to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()