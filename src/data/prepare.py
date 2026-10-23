from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_data():
    telemetry = pd.read_csv(RAW_DIR / "PdM_telemetry.csv")
    errors = pd.read_csv(RAW_DIR / "PdM_errors.csv")
    maint = pd.read_csv(RAW_DIR / "PdM_maint.csv")
    failures = pd.read_csv(RAW_DIR / "PdM_failures.csv")
    machines = pd.read_csv(RAW_DIR / "PdM_machines.csv")

    telemetry["datetime"] = pd.to_datetime(telemetry["datetime"])
    errors["datetime"] = pd.to_datetime(errors["datetime"])
    maint["datetime"] = pd.to_datetime(maint["datetime"])
    failures["datetime"] = pd.to_datetime(failures["datetime"])

    return telemetry, errors, maint, failures, machines


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
    Verify that every positive label really has a failure
    within the next 24 hours.
    """

    print("\nValidating labels...")

    positive = data[
        data["failure_next_24h"] == 1
    ].copy()

    failures = failures.sort_values(
        ["machineID", "datetime"]
    )

    # Find the next failure for each positive prediction timestamp.
    next_failure = pd.merge_asof(
        positive.sort_values("datetime"),
        failures[
            ["machineID", "datetime"]
        ].sort_values("datetime"),
        left_on="datetime",
        right_on="datetime",
        by="machineID",
        direction="forward",
        allow_exact_matches=False,
    )

    # The above merge has the same datetime column name on both sides,
    # so use a differently named failure table for clarity.

def main():
    telemetry, errors, maint, failures, machines = load_data()

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