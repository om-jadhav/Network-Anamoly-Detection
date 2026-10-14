
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "Friday-DDos-cleaned.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "Friday-DDos-validated.csv"


def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    original_rows = len(df)

    # Replace any remaining infinite values with NaN.
    numeric_columns = df.select_dtypes(include="number").columns
    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    # Remove rows with impossible negative flow duration.
    invalid_duration = df["Flow Duration"] < 0
    removed_duration = int(invalid_duration.sum())
    df = df.loc[~invalid_duration].copy()

    # Check negative values in inter-arrival-time features.
    iat_columns = [
        column for column in df.columns
        if "IAT" in column
    ]

    print("\nNegative IAT values after duration filtering:")
    for column in iat_columns:
        count = int((df[column] < 0).sum())
        if count:
            print(f"{column}: {count}")

    # Save validated data.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 45)
    print("VALIDATION SUMMARY")
    print("=" * 45)
    print("Original rows:", f"{original_rows:,}")
    print("Rows removed for negative duration:", removed_duration)
    print("Remaining rows:", f"{len(df):,}")
    print("Remaining missing values:", int(df.isna().sum().sum()))
    print("Remaining infinite values:", int(
        np.isinf(
            df.select_dtypes(include="number").to_numpy()
        ).sum()
    ))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
