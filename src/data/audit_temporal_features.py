from pathlib import Path

import pandas as pd


FEATURE_PATH = Path(
    "data/features/baseline_features.csv"
)

RAW_ERROR_PATH = Path(
    "data/raw/PdM_errors.csv"
)

TARGET = "failure_next_24h"

ERROR_COLUMNS = [
    "error1_count_24h",
    "error2_count_24h",
    "error3_count_24h",
    "error4_count_24h",
    "error5_count_24h",
    "total_errors_24h",
]


def main():
    print("Loading feature dataset...")
    features = pd.read_csv(FEATURE_PATH)

    features["datetime"] = pd.to_datetime(
        features["datetime"]
    )

    errors = pd.read_csv(RAW_ERROR_PATH)

    errors["datetime"] = pd.to_datetime(
        errors["datetime"]
    )

    print(
        f"Feature rows: {len(features):,}"
    )

    print(
        f"Error rows: {len(errors):,}"
    )

    # --------------------------------------------------
    # 1. Check feature datetime ordering
    # --------------------------------------------------

    print("\nChecking feature ordering...")

    ordered = features.sort_values(
        ["machineID", "datetime"]
    )

    duplicates = ordered.duplicated(
        ["machineID", "datetime"]
    ).sum()

    print(
        f"Duplicate machine-hour rows: {duplicates}"
    )

    # --------------------------------------------------
    # 2. Reconstruct error counts independently
    # --------------------------------------------------

    print(
        "\nReconstructing previous-24h error counts..."
    )

    errors = errors.sort_values(
        ["machineID", "datetime"]
    )

    # Count errors in the same 24 hourly observations
    # used by the feature generator:
    #
    # (t - 24h, t]
    #
    # This includes the current hour and the
    # previous 23 hours.
    reconstructed = []

    for machine_id, machine_errors in errors.groupby(
        "machineID"
    ):
        machine_features = features[
            features["machineID"] == machine_id
        ].copy()

        error_times = (
            machine_errors["datetime"]
            .sort_values()
            .reset_index(drop=True)
        )

        for _, row in machine_features.iterrows():

            t = row["datetime"]

            count = (
                (error_times <= t)
                & (error_times > t - pd.Timedelta(hours=24))
            ).sum()

            reconstructed.append(
                {
                    "machineID": machine_id,
                    "datetime": t,
                    "reconstructed_total_errors_24h": count,
                }
            )

    reconstructed = pd.DataFrame(
        reconstructed
    )

    # --------------------------------------------------
    # 3. Compare against feature
    # --------------------------------------------------

    comparison = features[
        [
            "machineID",
            "datetime",
            "total_errors_24h",
            TARGET,
        ]
    ].merge(
        reconstructed,
        on=["machineID", "datetime"],
        how="left",
    )

    comparison["difference"] = (
        comparison["total_errors_24h"]
        - comparison[
            "reconstructed_total_errors_24h"
        ]
    )

    print("\nError feature audit:")
    print(
        comparison["difference"]
        .describe()
    )

    max_difference = comparison[
        "difference"
    ].abs().max()

    print(
        f"\nMaximum absolute difference: "
        f"{max_difference}"
    )

    mismatches = comparison[
        comparison["difference"] != 0
    ]

    print(
        f"Mismatched rows: "
        f"{len(mismatches):,}"
    )

    if len(mismatches) > 0:
        print("\nFirst mismatches:")
        print(
            mismatches.head(20).to_string(
                index=False
            )
        )

    # --------------------------------------------------
    # 4. Check errors after prediction time
    # --------------------------------------------------

    print(
        "\nChecking future-error contamination..."
    )

    future_errors = []

    for machine_id, machine_errors in errors.groupby(
        "machineID"
    ):
        machine_features = features[
            features["machineID"] == machine_id
        ]

        error_times = (
            machine_errors["datetime"]
            .sort_values()
            .tolist()
        )

        for t in machine_features[
            "datetime"
        ].tolist():

            future_count = sum(
                error_time > t
                and error_time <= t + pd.Timedelta(hours=24)
                for error_time in error_times
            )

            if future_count > 0:
                future_errors.append(
                    {
                        "machineID": machine_id,
                        "datetime": t,
                        "future_errors": future_count,
                    }
                )

    print(
        f"Rows with errors occurring "
        f"in the following 24h: "
        f"{len(future_errors):,}"
    )

    print(
        "\nAudit complete."
    )


if __name__ == "__main__":
    main()