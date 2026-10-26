from pathlib import Path

import pandas as pd
import yaml


PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
FEATURE_DIR = Path("data/features")

SENSOR_COLUMNS = [
    "volt",
    "rotate",
    "pressure",
    "vibration",
]


def load_params():
    with open("params.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_data():
    machine_hourly = pd.read_csv(
        PROCESSED_DIR / "machine_hourly.csv"
    )

    errors = pd.read_csv(
        RAW_DIR / "PdM_errors.csv"
    )

    maintenance = pd.read_csv(
        RAW_DIR / "PdM_maint.csv"
    )

    machine_hourly["datetime"] = pd.to_datetime(
        machine_hourly["datetime"]
    )

    errors["datetime"] = pd.to_datetime(
        errors["datetime"]
    )

    maintenance["datetime"] = pd.to_datetime(
        maintenance["datetime"]
    )

    return machine_hourly, errors, maintenance


def add_telemetry_features(df, rolling_windows):
    """
    Add backward-looking rolling statistics for telemetry.

    Every rolling feature includes the current observation and
    previous observations only. No future timestamps are used.
    """

    df = df.sort_values(
        ["machineID", "datetime"]
    ).reset_index(drop=True)

    for window in rolling_windows:

        rolling = (
            df.groupby("machineID")[SENSOR_COLUMNS]
            .rolling(
                window=window,
                min_periods=window,
            )
        )

        rolling_mean = (
            rolling.mean()
            .reset_index(level=0, drop=True)
        )

        rolling_std = (
            rolling.std(ddof = 0)
            .reset_index(level=0, drop=True)
        )

        for sensor in SENSOR_COLUMNS:
            df[f"{sensor}_mean_{window}h"] = (
                rolling_mean[sensor]
            )

            df[f"{sensor}_std_{window}h"] = (
                rolling_std[sensor]
            )

    return df


def add_error_features(df, errors, error_window_hours):
    """
    Create historical error counts.

    Only errors occurring at or before the prediction timestamp
    are included.
    """

    # Count errors occurring at the same machine/timestamp/type.
    hourly_errors = (
        errors.groupby(
            ["machineID", "datetime", "errorID"]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Make the error columns explicit.
    expected_error_columns = [
        "error1",
        "error2",
        "error3",
        "error4",
        "error5",
    ]

    for error_type in expected_error_columns:
        if error_type not in hourly_errors.columns:
            hourly_errors[error_type] = 0

    hourly_errors = hourly_errors[
        ["machineID", "datetime"] + expected_error_columns
    ]

    # Merge exact hourly error counts onto telemetry.
    df = df.merge(
        hourly_errors,
        on=["machineID", "datetime"],
        how="left",
        validate="one_to_one",
    )

    df[expected_error_columns] = (
        df[expected_error_columns]
        .fillna(0)
    )

    # Calculate backward-looking rolling counts.
    for error_type in expected_error_columns:

        rolling_count = (
            df.groupby("machineID")[error_type]
            .rolling(
                window=error_window_hours,
                min_periods=1,
            )
            .sum()
            .reset_index(level=0, drop=True)
        )

        df[f"{error_type}_count_{error_window_hours}h"] = (
            rolling_count
        )

    count_columns = [
        f"{error_type}_count_{error_window_hours}h"
        for error_type in expected_error_columns
    ]

    df["total_errors_24h"] = df[count_columns].sum(axis=1)

    # Remove the intermediate hourly error columns.
    df = df.drop(
        columns=expected_error_columns
    )

    return df


def add_maintenance_features(df, maintenance):
    """
    Add time since the most recent maintenance event.

    The backward as-of merge guarantees that only maintenance
    events at or before the prediction timestamp are considered.
    """

    maintenance_history = (
        maintenance[
            ["machineID", "datetime"]
        ]
        .drop_duplicates(
            ["machineID", "datetime"]
        )
        .rename(
            columns={
                "datetime": "last_maintenance_datetime"
            }
        )
        .sort_values("last_maintenance_datetime")
    )

    df = df.sort_values("datetime")

    df = pd.merge_asof(
        df,
        maintenance_history,
        left_on="datetime",
        right_on="last_maintenance_datetime",
        by="machineID",
        direction="backward",
        allow_exact_matches=False,
    )

    df["hours_since_last_maintenance"] = (
        (
            df["datetime"]
            - df["last_maintenance_datetime"]
        )
        .dt.total_seconds()
        / (60 * 60)
    )

    df = df.drop(
        columns=["last_maintenance_datetime"]
    )

    return df


def encode_machine_model(df):
    """
    One-hot encode the machine model.
    """

    df = pd.get_dummies(
        df,
        columns=["model"],
        prefix="model",
        dtype="int8",
    )

    return df


def build_features():
    params = load_params()

    rolling_windows = params["features"]["rolling_windows"]
    error_window_hours = params["features"]["error_window_hours"]

    df, errors, maintenance = load_data()

    print("Initial shape:", df.shape)

    print("\nAdding telemetry features...")
    df = add_telemetry_features(
        df,
        rolling_windows,
    )

    print("Adding error features...")
    df = add_error_features(
        df,
        errors,
        error_window_hours,
    )

    print("Adding maintenance features...")
    df = add_maintenance_features(
        df,
        maintenance,
    )

    print("Encoding machine model...")
    df = encode_machine_model(df)

    # Rolling 48-hour features require 48 observations.
    # Remove the initial rows where these features do not exist.
    required_columns = [
        f"{sensor}_mean_48h"
        for sensor in SENSOR_COLUMNS
    ]

    before = len(df)

    df = df.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    removed = before - len(df)

    print(
        f"\nRemoved {removed} rows without a complete "
        "48-hour history."
    )

    df = df.sort_values(
        ["machineID", "datetime"]
    ).reset_index(drop=True)

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FEATURE_DIR / "baseline_features.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print("\nFeature dataset created:")
    print(output_path)

    print("\nFinal shape:")
    print(df.shape)

    print("\nFeature columns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(
        df.isna().sum()
        .sort_values(ascending=False)
        .head(20)
    )

    print("\nTarget distribution:")
    print(
        df["failure_next_24h"]
        .value_counts()
    )


if __name__ == "__main__":
    build_features()