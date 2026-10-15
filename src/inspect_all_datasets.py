
from pathlib import Path
from collections import Counter

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "MachineLearningCSV"


def inspect_dataset(file_path):
    print(f"\nInspecting: {file_path.name}")

    total_rows = 0
    label_counts = Counter()
    columns = None

    # Read in chunks to reduce memory usage.
    for chunk in pd.read_csv(
        file_path,
        chunksize=100_000,
        encoding="utf-8-sig",
        low_memory=False,
    ):
        chunk.columns = chunk.columns.str.strip()

        if columns is None:
            columns = chunk.columns.tolist()

        total_rows += len(chunk)

        if "Label" in chunk.columns:
            labels = chunk["Label"].astype("string").str.strip()
            label_counts.update(labels.value_counts().to_dict())

    print(f"Total records: {total_rows:,}")
    print(f"Total columns: {len(columns) if columns else 0}")

    print("Label distribution:")
    for label, count in label_counts.most_common():
        percentage = count / total_rows * 100
        print(f"  {label}: {count:,} ({percentage:.2f}%)")

    return set(columns or [])


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}"
        )

    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}"
        )

    print("=" * 65)
    print("CIC-IDS2017: ALL DATASET INSPECTION")
    print("=" * 65)
    print(f"Dataset folder: {DATA_DIR}")
    print(f"CSV files found: {len(csv_files)}")

    reference_columns = None
    compatible = True
    total_rows_all_files = 0

    for file_path in csv_files:
        columns = inspect_dataset(file_path)

        # Read only the header to compare column names.
        header = pd.read_csv(
            file_path,
            nrows=0,
            encoding="utf-8-sig",
        )
        normalized_columns = set(
            header.columns.str.strip().tolist()
        )

        if reference_columns is None:
            reference_columns = normalized_columns
        elif normalized_columns != reference_columns:
            compatible = False
            print("\nWARNING: Column mismatch detected!")
            print("Missing columns:", sorted(
                reference_columns - normalized_columns
            ))
            print("Extra columns:", sorted(
                normalized_columns - reference_columns
            ))

    print("\n" + "=" * 65)
    print("INSPECTION COMPLETE")
    print("=" * 65)
    print(f"Files inspected: {len(csv_files)}")
    print(
        "All files have matching column names:",
        "Yes" if compatible else "No",
    )


if __name__ == "__main__":
    main()
