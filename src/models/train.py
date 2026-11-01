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
MODEL_DIR = Path("models")
METRICS_DIR = Path("metrics")
SCHEMA_PATH = Path("src/serving/feature_schema.json")

TARGET = "failure_next_24h"

DROP_COLUMNS = [
    "datetime",
    "machineID",
    TARGET,
]


def load_params():
    import yaml

    with open("params.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_data():
    df = pd.read_csv(FEATURE_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values(["datetime", "machineID"]).reset_index(drop=True)


def split_data(df, params):
    train_end = params["split"]["train_end"]
    val_end = params["split"]["validation_end"]

    train = df[df["datetime"] < train_end].copy()
    validation = df[
        (df["datetime"] >= train_end) & (df["datetime"] < val_end)
    ].copy()
    test = df[df["datetime"] >= val_end].copy()

    return train, validation, test


def prepare_xy(df):
    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET]
    return X, y


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "threshold": float(threshold),
    }


def find_best_threshold(y_true, probabilities):
    """Select the threshold that maximises F1 on the validation set."""
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in [i / 100 for i in range(1, 100)]:
        metrics = calculate_metrics(y_true, probabilities, threshold)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold
    return best_threshold


def train_model(X_train, y_train, params):
    positives = y_train.sum()
    negatives = len(y_train) - positives
    scale_pos_weight = negatives / positives

    print(f"Training positives: {positives}")
    print(f"Training negatives: {negatives}")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    mp = params["models"]["baseline"]

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=mp["n_estimators"],
        max_depth=mp["max_depth"],
        learning_rate=mp["learning_rate"],
        subsample=mp["subsample"],
        colsample_bytree=mp["colsample_bytree"],
        min_child_weight=mp["min_child_weight"],
        reg_lambda=mp["reg_lambda"],
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=mp["random_state"],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, scale_pos_weight


def main():
    import mlflow
    import mlflow.xgboost

    print("Loading feature dataset...")
    params = load_params()
    df = load_data()

    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df['datetime'].min()} → {df['datetime'].max()}")

    train, validation, test = split_data(df, params)

    print("\nSplit sizes:")
    print(f"  Train:      {len(train):,}")
    print(f"  Validation: {len(validation):,}")
    print(f"  Test:       {len(test):,}")

    X_train, y_train = prepare_xy(train)
    X_val, y_val = prepare_xy(validation)
    X_test, y_test = prepare_xy(test)

    print(f"\nFeature count: {X_train.shape[1]}")
    print("\nPositive rates:")
    print(f"  Train:      {y_train.mean():.4%}")
    print(f"  Validation: {y_val.mean():.4%}")
    print(f"  Test:       {y_test.mean():.4%}")

    print("\nTraining XGBoost...")

    mlflow.set_experiment("predictive-maintenance")

    with mlflow.start_run(run_name="xgboost-baseline"):
        mp = params["models"]["baseline"]

        model, scale_pos_weight = train_model(X_train, y_train, params)

        mlflow.log_params({
            "model_type": "xgboost",
            "n_estimators": mp["n_estimators"],
            "max_depth": mp["max_depth"],
            "learning_rate": mp["learning_rate"],
            "min_child_weight": mp["min_child_weight"],
            "subsample": mp["subsample"],
            "colsample_bytree": mp["colsample_bytree"],
            "scale_pos_weight": round(scale_pos_weight, 2),
        })

        print("Training complete.\n")

        val_probabilities = model.predict_proba(X_val)[:, 1]
        threshold = find_best_threshold(y_val, val_probabilities)
        validation_metrics = calculate_metrics(y_val, val_probabilities, threshold)

        mlflow.log_metrics({
            "validation_pr_auc": validation_metrics["pr_auc"],
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "validation_f1": validation_metrics["f1"],
            "validation_threshold": validation_metrics["threshold"],
        })

        print("Validation results:")
        for k, v in validation_metrics.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

        print("\nEvaluating on test set...")
        test_probabilities = model.predict_proba(X_test)[:, 1]
        test_metrics = calculate_metrics(y_test, test_probabilities, threshold)

        mlflow.log_metrics({
            "test_pr_auc": test_metrics["pr_auc"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_threshold": test_metrics["threshold"],
        })

        print("\nTest results:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

        # ── Save model ───────────────────────────────────────────────
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "xgboost_baseline.json"
        model.save_model(model_path)

        mlflow.xgboost.log_model(
            model,
            artifact_path="xgboost_model",
            input_example=X_train.head(5),
        )

        # ── Save feature names ───────────────────────────────────────
        feature_path = MODEL_DIR / "xgboost_features.json"
        with open(feature_path, "w", encoding="utf-8") as f:
            json.dump(X_train.columns.tolist(), f, indent=2)
        mlflow.log_artifact(str(feature_path))

        # ── Save metrics ─────────────────────────────────────────────
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = METRICS_DIR / "baseline_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "validation": validation_metrics,
                "test": test_metrics,
                "split": params["split"],
            }, f, indent=2)
        mlflow.log_artifact(str(metrics_path))

        # ── Update feature_schema.json with the trained threshold ────
        if SCHEMA_PATH.exists():
            with open(SCHEMA_PATH, encoding="utf-8") as f:
                schema = json.load(f)
            schema["threshold"] = threshold
            with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
                json.dump(schema, f, indent=2)
            print(f"\nUpdated {SCHEMA_PATH} with threshold={threshold}")

    print("\nArtifacts created:")
    print(f"  {model_path}")
    print(f"  {feature_path}")
    print(f"  {metrics_path}")


if __name__ == "__main__":
    main()
