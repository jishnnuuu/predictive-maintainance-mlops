from pathlib import Path

import pandas as pd


ERROR_PATH = Path("data/raw/PdM_errors.csv")
FAILURE_PATH = Path("data/raw/PdM_failures.csv")


def main():

    print("Loading errors and failures...")

    errors = pd.read_csv(ERROR_PATH)
    failures = pd.read_csv(FAILURE_PATH)

    errors["datetime"] = pd.to_datetime(errors["datetime"])
    failures["datetime"] = pd.to_datetime(failures["datetime"])

    errors = errors.sort_values(
        ["datetime", "machineID"]
    )

    failures = failures.sort_values(
        ["datetime", "machineID"]
    )

    failures = failures.rename(
        columns={"datetime": "failure_time"}
    )

    # For each failure, find the latest error
    # from the same machine that occurred before it.
    result = pd.merge_asof(
        failures,
        errors,
        left_on="failure_time",
        right_on="datetime",
        by="machineID",
        direction="backward",
        tolerance=pd.Timedelta(hours=48),
    )

    result["hours_before_failure"] = (
        result["failure_time"]
        - result["datetime"]
    ).dt.total_seconds() / 3600

    bins = [
        0,
        1,
        6,
        12,
        24,
        48,
    ]

    labels = [
        "0-1h",
        "1-6h",
        "6-12h",
        "12-24h",
        "24-48h",
    ]

    result["timing_bucket"] = pd.cut(
        result["hours_before_failure"],
        bins=bins,
        labels=labels,
        right=True,
    )

    print("\n" + "=" * 70)
    print("ERROR TIMING BEFORE FAILURE")
    print("=" * 70)

    print(
        result[
            [
                "machineID",
                "failure_time",
                "failure",
                "errorID",
                "hours_before_failure",
                "timing_bucket",
            ]
        ].head(20).to_string(index=False)
    )

    print("\nTiming distribution:")

    print(
        result["timing_bucket"]
        .value_counts()
        .sort_index()
    )

    print("\nTiming percentages:")

    print(
        (
            result["timing_bucket"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2)
    )

    print(
        "\nMedian hours before failure:",
        result["hours_before_failure"].median(),
    )

    print(
        "Mean hours before failure:",
        result["hours_before_failure"].mean(),
    )


if __name__ == "__main__":
    main()