import pandas as pd


RAW_DIR = "data/raw"


def main():
    errors = pd.read_csv(f"{RAW_DIR}/PdM_errors.csv")
    failures = pd.read_csv(f"{RAW_DIR}/PdM_failures.csv")

    errors["datetime"] = pd.to_datetime(errors["datetime"])
    failures["datetime"] = pd.to_datetime(failures["datetime"])

    print("=" * 70)
    print("ERROR → FAILURE TEMPORAL AUDIT")
    print("=" * 70)

    # For every failure, find errors for the same machine
    # in the preceding 24 hours.
    rows = []

    for _, failure in failures.iterrows():
        machine = failure["machineID"]
        failure_time = failure["datetime"]

        machine_errors = errors[
            errors["machineID"].eq(machine)
            & (errors["datetime"] < failure_time)
            & (
                errors["datetime"]
                >= failure_time - pd.Timedelta(hours=24)
            )
        ]

        rows.append(
            {
                "machineID": machine,
                "failure_time": failure_time,
                "failure": failure["failure"],
                "errors_previous_24h": len(machine_errors),
            }
        )
    
    features = pd.read_csv(
        "data/features/baseline_features.csv"
    )

    print("\n" + "=" * 70)
    print("ERROR FEATURE DISTRIBUTION")
    print("=" * 70)

    for column in [
        "total_errors_24h",
        "error1_count_24h",
        "error2_count_24h",
        "error3_count_24h",
        "error4_count_24h",
        "error5_count_24h",
    ]:
        print(f"\n{column}")

        print(
            features.groupby("failure_next_24h")[column]
            .agg(["count", "mean", "median", "max"])
        )

    result = pd.DataFrame(rows)

    print("\nErrors in previous 24h before failures:")
    print(result["errors_previous_24h"].describe())

    print("\nFailures with at least one previous-24h error:")
    print(
        (result["errors_previous_24h"] > 0).mean()
    )

    print("\nSample:")
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()