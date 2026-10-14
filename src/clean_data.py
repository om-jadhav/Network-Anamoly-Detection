
from pathlib import Path

import numpy as np
import pandas as pd


# 1. Define dataset paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "MachineLearningCSV"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "Friday-DDos-cleaned.csv"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    # 2. Clean column names
    df.columns = df.columns.str.strip()

    print("\nOriginal dataset shape:", df.shape)

    # 3. Clean label text
    df["Label"] = df["Label"].astype("string").str.strip()

    # 4. Replace infinite values with missing values
    numeric_columns = df.select_dtypes(include="number").columns

    infinite_count = int(
        np.isinf(df[numeric_columns].to_numpy()).sum()
    )

    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    print("Infinite values replaced:", infinite_count)

    # 5. Remove duplicate records
    before_duplicates = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before_duplicates - len(df)

    print("Duplicate rows removed:", duplicates_removed)

    # 6. Remove rows containing missing values
    before_missing = len(df)
    df = df.dropna()
    missing_rows_removed = before_missing - len(df)

    print("Rows removed due to missing values:", missing_rows_removed)

    # 7. Save cleaned dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # 8. Display final summary
    print("\n" + "=" * 45)
    print("DATA CLEANING SUMMARY")
    print("=" * 45)

    print("Final dataset shape:", df.shape)
    print("\nFinal label distribution:")
    print(df["Label"].value_counts())

    print("\nRemaining missing values:", int(df.isna().sum().sum()))
    print("Output saved to:", OUTPUT_FILE)
    print("\nCleaning completed successfully!")


if __name__ == "__main__":
    main()
