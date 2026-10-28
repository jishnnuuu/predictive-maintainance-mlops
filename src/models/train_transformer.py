from pathlib import Path
import json
import math
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


FEATURE_PATH = Path(
    "data/features/baseline_features.csv"
)

MODEL_DIR = Path("models")
METRICS_DIR = Path("metrics")

TARGET = "failure_next_24h"

DROP_COLUMNS = [
    "datetime",
    "machineID",
    TARGET,
]


# ============================================================
# CONFIGURATION
# ============================================================

def load_params():
    with open(
        "params.yaml",
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA LOADING
# ============================================================

def load_data():

    df = pd.read_csv(FEATURE_PATH)

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    return df.sort_values(
        ["machineID", "datetime"]
    ).reset_index(drop=True)


# ============================================================
# TEMPORAL SPLIT
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

    return train, validation, test


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(df):

    X = df.drop(
        columns=DROP_COLUMNS
    )

    y = df[TARGET]

    return X, y


# ============================================================
# SEQUENCE DATASET
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

        # Build sequences independently for each machine.
        #
        # This prevents a sequence from crossing machine
        # boundaries.

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

        target = self.y[end_idx]

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
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len,
        dropout,
    ):

        super().__init__()

        self.dropout = nn.Dropout(
            dropout
        )

        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        pe = torch.zeros(
            max_len,
            d_model,
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe,
        )

    def forward(self, x):

        # x:
        # batch × sequence_length × d_model

        x = x + self.pe[
            :, :x.size(1), :
        ]

        return self.dropout(x)


# ============================================================
# TRANSFORMER CLASSIFIER
# ============================================================

class TransformerClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
        max_len,
    ):

        super().__init__()

        # Project the original feature vector into
        # Transformer embedding space.

        self.input_projection = nn.Linear(
            input_size,
            d_model,
        )

        self.position_encoding = (
            PositionalEncoding(
                d_model=d_model,
                max_len=max_len,
                dropout=dropout,
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        )

        self.classifier = nn.Linear(
            d_model,
            1,
        )

    def forward(self, x):

        # --------------------------------------------------
        # Input
        # --------------------------------------------------
        #
        # batch × 24 × 48
        #

        x = self.input_projection(x)

        # --------------------------------------------------
        # Positional encoding
        # --------------------------------------------------

        x = self.position_encoding(x)

        # --------------------------------------------------
        # Transformer encoder
        # --------------------------------------------------

        x = self.transformer(x)

        # --------------------------------------------------
        # Most recent timestep
        # --------------------------------------------------

        last_output = x[:, -1, :]

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        logits = self.classifier(
            last_output
        )

        return logits.squeeze(1)


# ============================================================
# METRICS
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
# PREDICTION
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
# MAIN
# ============================================================

def main():

    params = load_params()

    seed = params["seed"]

    set_seed(seed)

    # --------------------------------------------------
    # Transformer parameters
    # --------------------------------------------------

    transformer_params = params[
        "models"
    ]["transformer"]

    window_size = transformer_params[
        "window_hours"
    ]

    d_model = transformer_params[
        "d_model"
    ]

    nhead = transformer_params[
        "nhead"
    ]

    num_layers = transformer_params[
        "num_layers"
    ]

    dim_feedforward = transformer_params[
        "dim_feedforward"
    ]

    dropout = transformer_params[
        "dropout"
    ]

    # --------------------------------------------------
    # Training parameters
    # --------------------------------------------------

    batch_size = params[
        "training"
    ]["batch_size"]

    epochs = params[
        "training"
    ]["epochs"]

    learning_rate = params[
        "training"
    ]["learning_rate"
    ]

    patience = params[
        "training"
    ].get(
        "early_stopping_patience",
        3,
    )

    gradient_clip = params[
        "training"
    ].get(
        "gradient_clip",
        1.0,
    )

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    print(
        "Loading feature dataset..."
    )

    df = load_data()

    print(
        f"Total rows: {len(df):,}"
    )

    # --------------------------------------------------
    # Split
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Prepare features
    # --------------------------------------------------

    X_train, y_train = prepare_features(
        train
    )

    X_val, y_val = prepare_features(
        validation
    )

    X_test, y_test = prepare_features(
        test
    )

    # --------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Create sequences
    # --------------------------------------------------

    print(
        f"\nCreating {window_size}-hour "
        "sequences..."
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

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nUsing device: {device}"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = TransformerClassifier(
        input_size=X_train.shape[1],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=window_size,
    ).to(device)

    # --------------------------------------------------
    # Class imbalance
    # --------------------------------------------------

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
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

    print("\nTransformer configuration:")

    print(
        f"d_model:          {d_model}"
    )

    print(
        f"attention heads:  {nhead}"
    )

    print(
        f"encoder layers:   {num_layers}"
    )

    print(
        f"feed-forward dim:  {dim_feedforward}"
    )

    print(
        f"dropout:           {dropout}"
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    print(
        "\nTraining Transformer..."
    )

    best_val_pr_auc = -np.inf

    best_state = None

    epochs_without_improvement = 0

    val_targets = None
    val_probabilities = None

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)

            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            # Prevent exploding gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip,
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

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

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

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Loss: {epoch_loss:.6f} | "
            f"Validation PR-AUC: "
            f"{val_pr_auc:.6f}"
        )

        # --------------------------------------------------
        # Early stopping
        # --------------------------------------------------

        if val_pr_auc > best_val_pr_auc:

            best_val_pr_auc = val_pr_auc

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

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

            if (
                epochs_without_improvement
                >= patience
            ):

                print(
                    "\nEarly stopping triggered."
                )

                break

    # --------------------------------------------------
    # Restore best model
    # --------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

        print(
            "\nRestored best model with "
            f"validation PR-AUC: "
            f"{best_val_pr_auc:.6f}"
        )

    # Recalculate validation predictions
    # using the restored best model.

    val_targets, val_probabilities = (
        predict_model(
            model,
            val_loader,
            device,
        )
    )

    # --------------------------------------------------
    # Validation threshold
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Test
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / "transformer_baseline.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "input_size":
                X_train.shape[1],

            "d_model":
                d_model,

            "nhead":
                nhead,

            "num_layers":
                num_layers,

            "dim_feedforward":
                dim_feedforward,

            "dropout":
                dropout,

            "window_size":
                window_size,
        },
        model_path,
    )

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = {

        "validation":
            validation_metrics,

        "test":
            test_metrics,

        "model": {

            "type":
                "transformer",

            "window_hours":
                window_size,

            "d_model":
                d_model,

            "nhead":
                nhead,

            "num_layers":
                num_layers,

            "dim_feedforward":
                dim_feedforward,

            "dropout":
                dropout,
        },

        "training": {

            "batch_size":
                batch_size,

            "epochs":
                epochs,

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
    }

    metrics_path = (
        METRICS_DIR
        / "transformer_metrics.json"
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

    print(
        "\nArtifacts created:"
    )

    print(model_path)

    print(metrics_path)


if __name__ == "__main__":
    main()