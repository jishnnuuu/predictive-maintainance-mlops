from pathlib import Path
import json

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

FEATURES_PATH = Path(
    "models/xgboost_features.json"
)

METRICS_PATH = Path(
    "metrics/baseline_metrics.json"
)

TARGET = "failure_next_24h"

DROP_COLUMNS = [
    "datetime",
    "machineID",
    TARGET,
]


def load_data():
    df = pd.read_csv(FEATURE_PATH)

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    return df.sort_values(
        ["datetime", "machineID"]
    ).reset_index(drop=True)


def split_data(df):
    train = df[
        df["datetime"] < "2015-09-01"
    ]

    validation = df[
        (df["datetime"] >= "2015-09-01")
        & (df["datetime"] < "2015-11-01")
    ]

    test = df[
        df["datetime"] >= "2015-11-01"
    ]

    return train, validation, test


def prepare_xy(df):
    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET]

    return X, y


def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "threshold": threshold,
    }


def main():
    print("Loading feature dataset...")

    df = load_data()

    train, validation, test = split_data(df)

    _, y_val = prepare_xy(validation)
    X_test, y_test = prepare_xy(test)

    with open(FEATURES_PATH) as f:
        feature_names = json.load(f)

    # Load trained XGBoost model.
    model = xgb.XGBClassifier()

    model.load_model(
        MODEL_PATH
    )

    # Make absolutely sure serving/evaluation
    # uses the same feature ordering.
    X_test = X_test[feature_names]

    test_probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    # Threshold was selected using validation,
    # never using test.
    val_X, _ = prepare_xy(validation)
    val_X = val_X[feature_names]

    val_probabilities = (
        model.predict_proba(val_X)[:, 1]
    )

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in [
        i / 100
        for i in range(1, 100)
    ]:
        predictions = (
            val_probabilities >= threshold
        ).astype(int)

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        best_threshold,
    )

    metrics = {
        "validation": {
            "threshold": best_threshold,
            "f1": best_f1,
        },
        "test": test_metrics,
        "split": {
            "train_end": "2015-08-31",
            "validation_start": "2015-09-01",
            "validation_end": "2015-10-31",
            "test_start": "2015-11-01",
        },
    }

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        METRICS_PATH,
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    print("\nFinal test results:")

    for key, value in test_metrics.items():
        print(
            f"{key}: {value:.6f}"
            if isinstance(value, float)
            else f"{key}: {value}"
        )

    print(
        f"\nMetrics saved to {METRICS_PATH}"
    )


if __name__ == "__main__":
    main()