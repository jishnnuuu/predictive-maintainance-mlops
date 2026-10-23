from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")


def load_data():
    telemetry = pd.read_csv(RAW_DIR / "PdM_telemetry.csv")
    errors = pd.read_csv(RAW_DIR / "PdM_errors.csv")
    maint = pd.read_csv(RAW_DIR / "PdM_maint.csv")
    failures = pd.read_csv(RAW_DIR / "PdM_failures.csv")
    machines = pd.read_csv(RAW_DIR / "PdM_machines.csv")

    # Convert datetime columns immediately after loading.
    telemetry["datetime"] = pd.to_datetime(telemetry["datetime"])
    errors["datetime"] = pd.to_datetime(errors["datetime"])
    maint["datetime"] = pd.to_datetime(maint["datetime"])
    failures["datetime"] = pd.to_datetime(failures["datetime"])

    return telemetry, errors, maint, failures, machines


def inspect_time_range(name, df):
    if "datetime" not in df.columns:
        return

    print(f"\n{name}")
    print("-" * 50)
    print("Minimum datetime:", df["datetime"].min())
    print("Maximum datetime:", df["datetime"].max())
    print("Number of machines:", df["machineID"].nunique())


def inspect_machine_coverage(telemetry):
    print("\nTelemetry observations per machine")
    print("-" * 50)

    counts = telemetry.groupby("machineID").size()

    print(counts.describe())

    print("\nMachines with unexpected number of observations:")
    print(counts[counts != counts.median()])


def inspect_telemetry_frequency(telemetry):
    print("\nTelemetry frequency")
    print("-" * 50)

    telemetry = telemetry.sort_values(["machineID", "datetime"])

    time_diff = telemetry.groupby("machineID")["datetime"].diff()

    print(time_diff.value_counts().head(10))


def inspect_failures(failures):
    print("\nFailure statistics")
    print("-" * 50)

    print("Total failure events:", len(failures))
    print("Unique machines with failures:", failures["machineID"].nunique())

    print("\nFailures by component:")
    print(failures["failure"].value_counts())

    print("\nFailures by machine:")
    print(failures["machineID"].value_counts().describe())


def inspect_events(name, df, column):
    print(f"\n{name}")
    print("-" * 50)
    print(df[column].value_counts())


def main():
    telemetry, errors, maint, failures, machines = load_data()

    inspect_time_range("Telemetry", telemetry)
    inspect_time_range("Errors", errors)
    inspect_time_range("Maintenance", maint)
    inspect_time_range("Failures", failures)

    print("\nMachines")
    print("-" * 50)
    print("Number of machines:", len(machines))
    print("Machine IDs:", machines["machineID"].nunique())
    print("\nModels:")
    print(machines["model"].value_counts())

    inspect_machine_coverage(telemetry)
    inspect_telemetry_frequency(telemetry)
    inspect_failures(failures)

    inspect_events("Errors by type", errors, "errorID")
    inspect_events("Maintenance by component", maint, "comp")


if __name__ == "__main__":
    main()