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


FEATURE_PATH = Path("data/features/baseline_features.csv")
OUTPUT_PATH = Path("metrics/ablation_metrics.json")

TARGET = "failure_next_24h"

DROP_COLUMNS = [
    "datetime",
    "machineID",
    TARGET,
]


def load_data():
    df = pd.read_csv(FEATURE_PATH)

    df["datetime"] = pd.to_datetime(df["datetime"])

    return df.sort_values(
        ["datetime", "machineID"]
    ).reset_index(drop=True)


def split_data(df):
    train = df[
        df["datetime"] < "2015-09-01"
    ].copy()

    validation = df[
        (df["datetime"] >= "2015-09-01")
        & (df["datetime"] < "2015-11-01")
    ].copy()

    test = df[
        df["datetime"] >= "2015-11-01"
    ].copy()

    return train, validation, test


def get_feature_groups(df):
    all_features = [
        column
        for column in df.columns
        if column not in DROP_COLUMNS
    ]

    telemetry = [
        column
        for column in all_features
        if any(
            column.startswith(sensor)
            for sensor in [
                "volt",
                "rotate",
                "pressure",
                "vibration",
            ]
        )
        or column == "age"
    ]

    errors = [
        column
        for column in all_features
        if "error" in column
    ]

    maintenance = [
        column
        for column in all_features
        if "maintenance" in column
    ]

    return {
        "telemetry_only": telemetry,
        "telemetry_maintenance": (
            telemetry + maintenance
        ),
        "telemetry_errors": (
            telemetry + errors
        ),
        "all_features": all_features,
    }


def train_model(X_train, y_train):
    positives = y_train.sum()
    negatives = len(y_train) - positives

    scale_pos_weight = (
        negatives / positives
    )

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def evaluate(
    model,
    X,
    y,
    threshold=0.95,
):
    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "pr_auc": average_precision_score(
            y,
            probabilities,
        ),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
    }


def main():

    print("Loading feature dataset...")

    df = load_data()

    train, validation, test = split_data(df)

    feature_groups = get_feature_groups(df)

    results = {}

    for name, features in feature_groups.items():

        print("\n" + "=" * 60)
        print(f"EXPERIMENT: {name}")
        print("=" * 60)

        print(
            f"Number of features: {len(features)}"
        )

        X_train = train[features]
        y_train = train[TARGET]

        X_val = validation[features]
        y_val = validation[TARGET]

        X_test = test[features]
        y_test = test[TARGET]

        print("Training...")

        model = train_model(
            X_train,
            y_train,
        )

        print("Evaluating validation...")

        validation_metrics = evaluate(
            model,
            X_val,
            y_val,
        )

        print(
            f"Validation PR-AUC: "
            f"{validation_metrics['pr_auc']:.6f}"
        )

        print(
            f"Validation F1: "
            f"{validation_metrics['f1']:.6f}"
        )

        print("Evaluating test...")

        test_metrics = evaluate(
            model,
            X_test,
            y_test,
        )

        print(
            f"Test PR-AUC: "
            f"{test_metrics['pr_auc']:.6f}"
        )

        print(
            f"Test F1: "
            f"{test_metrics['f1']:.6f}"
        )

        results[name] = {
            "feature_count": len(features),
            "validation": validation_metrics,
            "test": test_metrics,
        }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print("\nAblation results saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()