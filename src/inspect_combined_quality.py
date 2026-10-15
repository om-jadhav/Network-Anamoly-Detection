
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "cicids2017_combined.csv"

CHUNK_SIZE = 100_000


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    total_rows = 0
    missing_counts = {}
    infinite_counts = {}
    negative_counts = {}
    non_numeric_counts = {}
    label_counts = {}

    print("Inspecting combined CIC-IDS2017 dataset...")
    print("This may take a few minutes.\n")

    for chunk_number, chunk in enumerate(
        pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE, encoding="utf-8-sig"),
        start=1,
    ):
        chunk.columns = chunk.columns.str.strip()
        total_rows += len(chunk)

        if "Label" not in chunk.columns:
            raise ValueError("The dataset does not contain a Label column.")

        # Count labels
        labels = chunk["Label"].value_counts(dropna=False)
        for label, count in labels.items():
            label_counts[str(label)] = (
                label_counts.get(str(label), 0) + int(count)
            )

        # Inspect all feature columns
        for column in chunk.columns:
            if column == "Label":
                continue

            original = chunk[column]
            numeric = pd.to_numeric(original, errors="coerce")

            missing = int(numeric.isna().sum())
            non_numeric = int(
                (numeric.isna() & original.notna()).sum()
            )
            infinite = int(np.isinf(numeric).sum())
            negative = int((numeric < 0).sum())

            if missing:
                missing_counts[column] = (
                    missing_counts.get(column, 0) + missing
                )

            if non_numeric:
                non_numeric_counts[column] = (
                    non_numeric_counts.get(column, 0) + non_numeric
                )

            if infinite:
                infinite_counts[column] = (
                    infinite_counts.get(column, 0) + infinite
                )

            if negative:
                negative_counts[column] = (
                    negative_counts.get(column, 0) + negative
                )

        print(
            f"Processed chunk {chunk_number} | "
            f"Rows checked: {total_rows:,}"
        )

    print("\n" + "=" * 55)
    print("COMBINED DATASET QUALITY REPORT")
    print("=" * 55)
    print(f"Total rows checked: {total_rows:,}")
    print(f"Feature columns: 78")
    print(f"Unique labels: {len(label_counts)}")

    def show_counts(title, counts):
        print(f"\n{title}")
        if not counts:
            print("None found.")
        else:
            for column, count in sorted(
                counts.items(), key=lambda item: item[1], reverse=True
            ):
                print(f"{column}: {count:,}")

    show_counts("Missing values:", missing_counts)
    show_counts("Infinite values:", infinite_counts)
    show_counts("Non-numeric feature values:", non_numeric_counts)
    show_counts("Negative numeric values (investigate; not all are invalid):",
                negative_counts)

    print("\nLabel distribution:")
    for label, count in sorted(
        label_counts.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"{label}: {count:,}")

    print("\nInspection complete. No data was modified.")


if __name__ == "__main__":
    main()
