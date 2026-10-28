from pathlib import Path
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
import yaml

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from torch import nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Paths
# ============================================================

FEATURE_PATH = Path(
    "data/features/baseline_features.csv"
)

MODEL_DIR = Path("models")
METRICS_DIR = Path("metrics")


# ============================================================
# Configuration
# ============================================================

TARGET = "failure_next_24h"

DROP_COLUMNS = [
    "datetime",
    "machineID",
    TARGET,
]


# ============================================================
# Utility Functions
# ============================================================

def load_params():
    with open(
        "params.yaml",
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data Loading
# ============================================================

def load_data():

    df = pd.read_csv(
        FEATURE_PATH
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    return df.sort_values(
        ["machineID", "datetime"]
    ).reset_index(drop=True)


# ============================================================
# Temporal Split
# ============================================================

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

    return (
        train,
        validation,
        test,
    )


# ============================================================
# Feature Preparation
# ============================================================

def prepare_features(df):

    X = df.drop(
        columns=DROP_COLUMNS
    )

    y = df[TARGET]

    return X, y


# ============================================================
# Sequence Dataset
# ============================================================

class SequenceDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        machine_ids,
        window_size,
    ):

        self.X = X
        self.y = y
        self.machine_ids = machine_ids
        self.window_size = window_size

        self.indices = []

        # ----------------------------------------------------
        # Build valid sequence endpoints.
        #
        # Each sequence:
        #
        # [t-23, ..., t]
        #
        # contains observations from ONLY one machine.
        # ----------------------------------------------------

        for machine_id in machine_ids.unique():

            machine_indices = np.where(
                machine_ids.values == machine_id
            )[0]

            for i in range(
                window_size - 1,
                len(machine_indices),
            ):

                end_idx = machine_indices[i]

                start_idx = machine_indices[
                    i - window_size + 1
                ]

                self.indices.append(
                    (
                        start_idx,
                        end_idx,
                    )
                )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        start_idx, end_idx = self.indices[idx]

        sequence = self.X[
            start_idx:end_idx + 1
        ]

        target = self.y[
            end_idx
        ]

        return (
            torch.tensor(
                sequence,
                dtype=torch.float32,
            ),
            torch.tensor(
                target,
                dtype=torch.float32,
            ),
        )


# ============================================================
# LSTM Model
# ============================================================

class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.classifier = nn.Linear(
            hidden_size,
            1,
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        # ----------------------------------------------------
        # Use the representation corresponding to the
        # most recent hour.
        # ----------------------------------------------------

        last_output = output[:, -1, :]

        logits = self.classifier(
            last_output
        )

        return logits.squeeze(1)


# ============================================================
# Metrics
# ============================================================

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


# ============================================================
# Threshold Selection
# ============================================================

def find_best_threshold(
    y_true,
    probabilities,
):

    best_threshold = 0.5
    best_f1 = 0.0

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


# ============================================================
# Prediction
# ============================================================

def predict_model(
    model,
    loader,
    device,
):

    model.eval()

    probabilities = []
    targets = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)

            logits = model(
                X_batch
            )

            probs = torch.sigmoid(
                logits
            )

            probabilities.extend(
                probs.cpu().numpy()
            )

            targets.extend(
                y_batch.numpy()
            )

    return (
        np.array(targets),
        np.array(probabilities),
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load parameters
    # --------------------------------------------------------

    params = load_params()

    seed = params["seed"]

    set_seed(seed)

    window_size = params[
        "models"
    ]["sequence"]["window_hours"]

    hidden_size = params[
        "models"
    ]["sequence"]["hidden_size"]

    num_layers = params[
        "models"
    ]["sequence"]["num_layers"]

    dropout = params[
        "models"
    ]["sequence"]["dropout"]

    batch_size = params[
        "training"
    ]["batch_size"]

    epochs = params[
        "training"
    ]["epochs"]

    learning_rate = params[
        "training"
    ]["learning_rate"]

    # --------------------------------------------------------
    # Optional training parameters
    # --------------------------------------------------------

    # If these parameters do not exist in params.yaml,
    # the defaults below are used.

    patience = params.get(
        "training",
        {}
    ).get(
        "early_stopping_patience",
        3,
    )

    gradient_clip = params.get(
        "training",
        {}
    ).get(
        "gradient_clip",
        1.0,
    )

    num_workers = params.get(
        "training",
        {}
    ).get(
        "num_workers",
        0,
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        "Loading feature dataset..."
    )

    df = load_data()

    print(
        f"Total rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # Temporal split
    # --------------------------------------------------------

    train, validation, test = split_data(
        df
    )

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

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X_train, y_train = prepare_features(
        train
    )

    X_val, y_val = prepare_features(
        validation
    )

    X_test, y_test = prepare_features(
        test
    )

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    X_train = X_train.to_numpy(
        dtype=np.float32
    )

    X_val = X_val.to_numpy(
        dtype=np.float32
    )

    X_test = X_test.to_numpy(
        dtype=np.float32
    )

    y_train = y_train.to_numpy(
        dtype=np.float32
    )

    y_val = y_val.to_numpy(
        dtype=np.float32
    )

    y_test = y_test.to_numpy(
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Machine IDs
    # --------------------------------------------------------

    train_machine_ids = (
        train["machineID"]
        .reset_index(drop=True)
    )

    val_machine_ids = (
        validation["machineID"]
        .reset_index(drop=True)
    )

    test_machine_ids = (
        test["machineID"]
        .reset_index(drop=True)
    )

    print(
        f"\nNumber of features: "
        f"{X_train.shape[1]}"
    )

    # --------------------------------------------------------
    # Create sequences
    # --------------------------------------------------------

    print(
        f"\nCreating {window_size}-hour sequences..."
    )

    train_dataset = SequenceDataset(
        X_train,
        y_train,
        train_machine_ids,
        window_size,
    )

    val_dataset = SequenceDataset(
        X_val,
        y_val,
        val_machine_ids,
        window_size,
    )

    test_dataset = SequenceDataset(
        X_test,
        y_test,
        test_machine_ids,
        window_size,
    )

    print(
        f"Train sequences: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation sequences: "
        f"{len(val_dataset):,}"
    )

    print(
        f"Test sequences: "
        f"{len(test_dataset):,}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nUsing device: {device}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = LSTMClassifier(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    positives = y_train.sum()

    negatives = (
        len(y_train)
        - positives
    )

    pos_weight = torch.tensor(
        [negatives / positives],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    print(
        f"\nPositive samples: "
        f"{int(positives):,}"
    )

    print(
        f"Negative samples: "
        f"{int(negatives):,}"
    )

    print(
        f"Positive weight: "
        f"{pos_weight.item():.2f}"
    )

    print(
        f"Early stopping patience: "
        f"{patience}"
    )

    print(
        f"Gradient clipping: "
        f"{gradient_clip}"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print(
        "\nTraining LSTM..."
    )

    best_val_pr_auc = -np.inf

    best_state_dict = None

    epochs_without_improvement = 0

    training_history = []

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        # ----------------------------------------------------
        # Training batches
        # ----------------------------------------------------

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            # ------------------------------------------------
            # Gradient clipping
            #
            # Helps prevent unstable gradients in recurrent
            # neural networks.
            # ------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip,
            )

            optimizer.step()

            running_loss += (
                loss.item()
                * len(X_batch)
            )

        epoch_loss = (
            running_loss
            / len(train_dataset)
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_targets, val_probabilities = (
            predict_model(
                model,
                val_loader,
                device,
            )
        )

        val_pr_auc = (
            average_precision_score(
                val_targets,
                val_probabilities,
            )
        )

        training_history.append(
            {
                "epoch": epoch + 1,
                "loss": float(
                    epoch_loss
                ),
                "validation_pr_auc": float(
                    val_pr_auc
                ),
            }
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Loss: {epoch_loss:.6f} | "
            f"Validation PR-AUC: "
            f"{val_pr_auc:.6f}"
        )

        # ----------------------------------------------------
        # Best model tracking
        # ----------------------------------------------------

        if val_pr_auc > best_val_pr_auc:

            best_val_pr_auc = val_pr_auc

            best_state_dict = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

            print(
                "  -> New best model"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"  -> No improvement "
                f"({epochs_without_improvement}/"
                f"{patience})"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_state_dict is not None:

        model.load_state_dict(
            best_state_dict
        )

        print(
            f"\nRestored best model "
            f"with validation PR-AUC: "
            f"{best_val_pr_auc:.6f}"
        )

    # --------------------------------------------------------
    # Final validation prediction
    # --------------------------------------------------------

    val_targets, val_probabilities = (
        predict_model(
            model,
            val_loader,
            device,
        )
    )

    # --------------------------------------------------------
    # Validation threshold
    # --------------------------------------------------------

    threshold = find_best_threshold(
        val_targets,
        val_probabilities,
    )

    validation_metrics = calculate_metrics(
        val_targets,
        val_probabilities,
        threshold,
    )

    print(
        "\nValidation results:"
    )

    for key, value in validation_metrics.items():

        print(
            f"{key}: {value:.6f}"
            if isinstance(value, float)
            else f"{key}: {value}"
        )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print(
        "\nEvaluating test set..."
    )

    test_targets, test_probabilities = (
        predict_model(
            model,
            test_loader,
            device,
        )
    )

    test_metrics = calculate_metrics(
        test_targets,
        test_probabilities,
        threshold,
    )

    print(
        "\nTest results:"
    )

    for key, value in test_metrics.items():

        print(
            f"{key}: {value:.6f}"
            if isinstance(value, float)
            else f"{key}: {value}"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / "lstm_baseline.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "input_size":
                X_train.shape[1],

            "hidden_size":
                hidden_size,

            "num_layers":
                num_layers,

            "dropout":
                dropout,

            "window_size":
                window_size,

            "best_validation_pr_auc":
                float(best_val_pr_auc),

            "threshold":
                float(threshold),
        },
        model_path,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = {

        "validation":
            validation_metrics,

        "test":
            test_metrics,

        "best_validation_pr_auc":
            float(best_val_pr_auc),

        "model": {

            "type":
                "lstm",

            "window_hours":
                window_size,

            "hidden_size":
                hidden_size,

            "num_layers":
                num_layers,

            "dropout":
                dropout,
        },

        "training": {

            "batch_size":
                batch_size,

            "epochs_requested":
                epochs,

            "epochs_completed":
                len(training_history),

            "learning_rate":
                learning_rate,

            "pos_weight":
                float(
                    pos_weight.item()
                ),

            "early_stopping_patience":
                patience,

            "gradient_clip":
                gradient_clip,
        },

        "history":
            training_history,
    }

    metrics_path = (
        METRICS_DIR
        / "lstm_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\nArtifacts created:"
    )

    print(
        model_path
    )

    print(
        metrics_path
    )


if __name__ == "__main__":
    main()