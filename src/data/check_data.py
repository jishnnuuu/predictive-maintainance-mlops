from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")


def inspect_csv(path: Path) -> None:
    print("\n" + "=" * 80)
    print(f"FILE: {path.name}")
    print("=" * 80)

    df = pd.read_csv(path)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nInfo:")
    df.info()

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())


def main() -> None:
    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in data/raw/. "
            "Download the dataset and place the CSV files there."
        )

    print(f"Found {len(csv_files)} CSV files.")

    for path in csv_files:
        inspect_csv(path)


if __name__ == "__main__":
    main()