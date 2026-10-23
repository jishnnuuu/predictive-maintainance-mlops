from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_data():
    data = pd.read_csv(
        PROCESSED_DIR / "machine_hourly.csv"
    )

    failures = pd.read_csv(
        RAW_DIR / "PdM_failures.csv"
    )

    data["datetime"] = pd.to_datetime(data["datetime"])
    failures["datetime"] = pd.to_datetime(failures["datetime"])

    return data, failures


def validate_labels(data, failures):
    """
    Verify that failure_next_24h exactly matches the existence
    of a failure in the interval:

        (t, t + 24 hours]
    """

    failure_times = failures[
        ["machineID", "datetime"]
    ].rename(
        columns={"datetime": "failure_datetime"}
    )

    # Find the first failure strictly after each prediction time.
    checked = pd.merge_asof(
        data.sort_values("datetime"),
        failure_times.sort_values("failure_datetime"),
        left_on="datetime",
        right_on="failure_datetime",
        by="machineID",
        direction="forward",
        allow_exact_matches=False,
    )

    time_to_failure = (
        checked["failure_datetime"]
        - checked["datetime"]
    )

    expected_label = (
        time_to_failure.notna()
        & (time_to_failure <= pd.Timedelta(hours=24))
    ).astype("int8")

    mismatch = (
        expected_label != checked["failure_next_24h"]
    )

    print("=" * 70)
    print("LABEL VALIDATION")
    print("=" * 70)

    print("\nTotal rows:", len(data))

    print(
        "Expected positive labels:",
        expected_label.sum(),
    )

    print(
        "Actual positive labels:",
        data["failure_next_24h"].sum(),
    )

    print(
        "Mismatched labels:",
        mismatch.sum(),
    )

    if mismatch.sum() == 0:
        print("\n✓ LABEL VALIDATION PASSED")
    else:
        print("\n✗ LABEL VALIDATION FAILED")

        print("\nFirst mismatches:")
        print(
            checked.loc[
                mismatch,
                [
                    "machineID",
                    "datetime",
                    "failure_datetime",
                    "failure_next_24h",
                ],
            ].head(20)
        )

        raise AssertionError(
            "Failure labels do not match the expected "
            "24-hour forward-looking definition."
        )


def main():
    data, failures = load_data()

    validate_labels(
        data,
        failures,
    )


if __name__ == "__main__":
    main()