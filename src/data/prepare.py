from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_data():
    telemetry = pd.read_csv(
        RAW_DIR / "PdM_telemetry.csv"
    )

    failures = pd.read_csv(
        RAW_DIR / "PdM_failures.csv"
    )

    machines = pd.read_csv(
        RAW_DIR / "PdM_machines.csv"
    )

    telemetry["datetime"] = pd.to_datetime(
        telemetry["datetime"]
    )

    failures["datetime"] = pd.to_datetime(
        failures["datetime"]
    )

    return telemetry, failures, machines


def create_failure_label(telemetry, failures):
    """
    Create a binary label indicating whether a machine failure
    occurs in the next 24 hours.

    Definition:

        y(t) = 1 if t < failure_time <= t + 24h
        y(t) = 0 otherwise
    """

    telemetry = telemetry.sort_values(
        ["machineID", "datetime"]
    ).copy()

    failures = failures.sort_values(
        ["machineID", "datetime"]
    ).copy()

    # Rename the failure timestamp so it is clear that this
    # timestamp represents a future event relative to prediction_time.
    future_failures = failures[
        ["machineID", "datetime"]
    ].rename(
        columns={"datetime": "failure_datetime"}
    )

    # For every telemetry timestamp, find the first failure
    # occurring after that timestamp for the same machine.
    labeled = pd.merge_asof(
        telemetry.sort_values("datetime"),
        future_failures.sort_values("failure_datetime"),
        left_on="datetime",
        right_on="failure_datetime",
        by="machineID",
        direction="forward",
        allow_exact_matches=False,
    )

    # Calculate time until the next failure.
    time_to_failure = (
        labeled["failure_datetime"] - labeled["datetime"]
    )

    # Failure occurs strictly after t and within 24 hours.
    labeled["failure_next_24h"] = (
        time_to_failure.notna()
        & (time_to_failure <= pd.Timedelta(hours=24))
    ).astype("int8")

    # The failure timestamp was only an intermediate column.
    # It must NOT be present in the model features.
    labeled = labeled.drop(columns=["failure_datetime"])

    return labeled


def add_machine_metadata(telemetry, machines):
    return telemetry.merge(
        machines,
        on="machineID",
        how="left",
        validate="many_to_one",
    )

def validate_labels(data, failures):
    """
    Independently verify that every positive label has
    a failure strictly after the prediction timestamp
    and within the next 24 hours.
    """

    print("\nValidating labels...")

    failures_for_merge = (
        failures[
            ["machineID", "datetime"]
        ]
        .rename(
            columns={
                "datetime": "failure_datetime"
            }
        )
        .sort_values(
            ["failure_datetime", "machineID"]
        )
    )

    positive = data[
        data["failure_next_24h"] == 1
    ].copy()

    positive = positive.sort_values(
        ["datetime", "machineID"]
    )

    next_failure = pd.merge_asof(
        positive,
        failures_for_merge,
        left_on="datetime",
        right_on="failure_datetime",
        by="machineID",
        direction="forward",
        allow_exact_matches=False,
    )

    time_to_failure = (
        next_failure["failure_datetime"]
        - next_failure["datetime"]
    )

    valid_positive = (
        time_to_failure.notna()
        & (time_to_failure > pd.Timedelta(0))
        & (time_to_failure <= pd.Timedelta(hours=24))
    )

    print(
        f"Positive labels: {len(positive):,}"
    )

    print(
        f"Valid positive labels: "
        f"{valid_positive.sum():,}"
    )

    print(
        f"Invalid positive labels: "
        f"{(~valid_positive).sum():,}"
    )

    if (~valid_positive).any():
        print("\nInvalid positive examples:")
        print(
            next_failure.loc[
                ~valid_positive,
                [
                    "machineID",
                    "datetime",
                    "failure_datetime",
                ],
            ]
            .head(20)
            .to_string(index=False)
        )

    # Also check negative labels.
    negative = data[
        data["failure_next_24h"] == 0
    ].copy()

    negative = negative.sort_values(
        ["datetime", "machineID"]
    )

    next_negative_failure = pd.merge_asof(
        negative,
        failures_for_merge,
        left_on="datetime",
        right_on="failure_datetime",
        by="machineID",
        direction="forward",
        allow_exact_matches=False,
    )

    negative_time_to_failure = (
        next_negative_failure["failure_datetime"]
        - next_negative_failure["datetime"]
    )

    invalid_negative = (
        negative_time_to_failure.notna()
        & (
            negative_time_to_failure
            <= pd.Timedelta(hours=24)
        )
    )

    print(
        f"\nNegative labels: {len(negative):,}"
    )

    print(
        f"Invalid negative labels: "
        f"{invalid_negative.sum():,}"
    )

    if invalid_negative.any():
        print("\nInvalid negative examples:")
        print(
            next_negative_failure.loc[
                invalid_negative,
                [
                    "machineID",
                    "datetime",
                    "failure_datetime",
                ],
            ]
            .head(20)
            .to_string(index=False)
        )

    print("\nLabel validation complete.")

    return (
        (~valid_positive).sum() == 0
        and invalid_negative.sum() == 0
    )

def main():
    telemetry, failures, machines = load_data()

    print("Creating failure labels...")

    data = create_failure_label(
        telemetry,
        failures,
    )

    print("Adding machine metadata...")

    data = add_machine_metadata(
        data,
        machines,
    )
    
    validate_labels(
        data,
        failures,
    )

    data = data.sort_values(
        ["machineID", "datetime"]
    ).reset_index(drop=True)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = PROCESSED_DIR / "machine_hourly.csv"

    data.to_csv(
        output_path,
        index=False,
    )

    print("\nProcessed dataset created:")
    print(output_path)

    print("\nShape:")
    print(data.shape)

    print("\nColumns:")
    print(data.columns.tolist())

    print("\nTarget distribution:")
    print(data["failure_next_24h"].value_counts())

    print("\nTarget percentage:")
    print(
        data["failure_next_24h"]
        .value_counts(normalize=True)
        .mul(100)
    )


if __name__ == "__main__":
    main()