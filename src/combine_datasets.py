
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "MachineLearningCSV"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "cicids2017_combined.csv"
CHUNK_SIZE = 100_000

# Normalize labels that may contain a replacement character.
LABEL_MAP = {
    "Web Attack � Brute Force": "Web Attack - Brute Force",
    "Web Attack � XSS": "Web Attack - XSS",
    "Web Attack � Sql Injection": "Web Attack - SQL Injection",
}


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}"
        )

    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if len(csv_files) != 8:
        print(
            f"Warning: Expected 8 CSV files, found {len(csv_files)}."
        )

    if not csv_files:
        raise FileNotFoundError("No CSV files found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Avoid accidentally appending to an older combined dataset.
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    total_rows = 0
    first_chunk = True
    combined_labels = {}

    for file_path in csv_files:
        print(f"\nProcessing: {file_path.name}")

        file_rows = 0

        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            encoding="utf-8-sig",
            low_memory=False,
        ):
            # Standardize column names.
            chunk.columns = chunk.columns.str.strip()

            if "Label" not in chunk.columns:
                raise ValueError(
                    f"Missing Label column in {file_path.name}"
                )

            # Standardize label whitespace and known Web Attack labels.
            chunk["Label"] = (
                chunk["Label"]
                .astype("string")
                .str.strip()
                .replace(LABEL_MAP)
            )

            # Save the first chunk with headers; append subsequent chunks.
            chunk.to_csv(
                OUTPUT_FILE,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
                encoding="utf-8",
            )

            first_chunk = False

            file_rows += len(chunk)
            total_rows += len(chunk)

            counts = chunk["Label"].value_counts()
            for label, count in counts.items():
                combined_labels[label] = (
                    combined_labels.get(label, 0) + int(count)
                )

        print(f"Records added: {file_rows:,}")

    print("\n" + "=" * 55)
    print("COMBINATION COMPLETE")
    print("=" * 55)
    print(f"Files processed: {len(csv_files)}")
    print(f"Total records: {total_rows:,}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nCombined label distribution:")
    for label, count in sorted(
        combined_labels.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{label}: {count:,}")

    print("\nNo rows were intentionally removed.")
    print("Duplicate and missing-value checks come next.")


if __name__ == "__main__":
    main()
