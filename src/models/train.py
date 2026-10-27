from pathlib import Path
import json

import pandas as pd
import yaml
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


FEATURE_PATH = Path("data/features/baseline_features.csv")
MODEL_DIR = Path("models")

TARGET = "failure_next_24h"

DROP_COLUMNS = [
    "datetime",
    "machineID",
    TARGET,
]


def load_params():
    with open("params.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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


def prepare_xy(df):
    X = df.drop(
        columns=DROP_COLUMNS
    )

    y = df[TARGET]

    return X, y


def calculate_metrics(y_true, probabilities, threshold):
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


def find_best_threshold(y_true, probabilities):
    """
    Select the threshold that maximizes F1
    on the validation set.

    The test set is NOT used for threshold selection.
    """

    best_threshold = 0.5
    best_f1 = 0.0

    # Search thresholds across the useful probability range.
    for threshold in [
        i / 100
        for i in range(1, 100)
    ]:
        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold,
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold

    return best_threshold


def train_model(X_train, y_train, params):
    positives = y_train.sum()
    negatives = len(y_train) - positives

    scale_pos_weight = negatives / positives

    print(
        f"Training positives: {positives}"
    )
    print(
        f"Training negatives: {negatives}"
    )
    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.2f}"
    )

    model_params = params["models"]["baseline"]

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        learning_rate=model_params["learning_rate"],
        subsample=model_params["subsample"],
        colsample_bytree=model_params["colsample_bytree"],
        min_child_weight=model_params["min_child_weight"],
        reg_lambda=model_params["reg_lambda"],
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=model_params["random_state"],
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model

def main():
    print("Loading feature dataset...")

    df = load_data()

    print(
        f"Total rows: {len(df):,}"
    )

    print(
        f"Date range: "
        f"{df['datetime'].min()} → "
        f"{df['datetime'].max()}"
    )

    train, validation, test = split_data(df)

    print("\nSplit sizes:")
    print(
        f"Train:      {len(train):,}"
    )
    print(
        f"Validation: {len(validation):,}"
    )
    print(
        f"Test:       {len(test):,}"
    )

    X_train, y_train = prepare_xy(train)
    X_val, y_val = prepare_xy(validation)
    X_test, y_test = prepare_xy(test)

    print("\nFeature count:")
    print(X_train.shape[1])

    print("\nPositive rates:")
    print(
        f"Train:      {y_train.mean():.4%}"
    )
    print(
        f"Validation: {y_val.mean():.4%}"
    )
    print(
        f"Test:       {y_test.mean():.4%}"
    )

    print("\nTraining XGBoost...")
    
    params = load_params()

    model = train_model(
        X_train,
        y_train,
        params
    )

    print("Training complete.")

    print("\nGenerating validation predictions...")

    val_probabilities = model.predict_proba(
        X_val
    )[:, 1]

    validation_pr_auc = (
        average_precision_score(
            y_val,
            val_probabilities,
        )
    )

    threshold = find_best_threshold(
        y_val,
        val_probabilities,
    )

    validation_metrics = calculate_metrics(
        y_val,
        val_probabilities,
        threshold,
    )

    print("\nValidation results:")
    for key, value in validation_metrics.items():
        print(
            f"{key}: {value:.6f}"
            if isinstance(value, float)
            else f"{key}: {value}"
        )

    print("\nEvaluating on test set...")

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        threshold,
    )

    print("\nTest results:")
    for key, value in test_metrics.items():
        print(
            f"{key}: {value:.6f}"
            if isinstance(value, float)
            else f"{key}: {value}"
        )

    # Save model.
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR / "xgboost_baseline.json"
    )

    model.save_model(model_path)

    # Save feature names separately so serving
    # can enforce the exact training schema later.
    feature_path = (
        MODEL_DIR / "xgboost_features.json"
    )

    with open(
        feature_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            X_train.columns.tolist(),
            file,
            indent=2,
        )

    print("\nArtifacts created:")
    print(model_path)
    print(feature_path)


if __name__ == "__main__":
    main()