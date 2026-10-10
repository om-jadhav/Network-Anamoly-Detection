
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

# Locate the project data folder
DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "MachineLearningCSV"
)

FILE_NAME = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
FILE_PATH = DATA_DIR / FILE_NAME

if not FILE_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {FILE_PATH}")

print("Analyzing:", FILE_NAME)
print("Reading data in chunks to reduce memory usage...\n")

# Statistics collected across the entire CSV
total_rows = 0
label_counts = Counter()
missing_counts = Counter()
infinite_counts = Counter()
duplicate_rows_within_chunks = 0
columns = None

# Read 100,000 rows at a time instead of loading everything at once
for chunk in pd.read_csv(
    FILE_PATH,
    chunksize=100_000,
    encoding="utf-8-sig"
):
    # CIC-IDS2017 column names may contain leading/trailing spaces
    chunk.columns = chunk.columns.str.strip()

    if columns is None:
        columns = chunk.columns.tolist()

    total_rows += len(chunk)

    # Count normal and attack labels
    labels = chunk["Label"].astype("string").str.strip()
    label_counts.update(labels.value_counts().to_dict())

    # Count missing values
    missing_counts.update(
        chunk.isna().sum().to_dict()
    )

    # Count positive and negative infinity in numeric columns
    numeric = chunk.select_dtypes(include="number")
    infinite_mask = pd.DataFrame(
        np.isinf(numeric.to_numpy()),
        columns=numeric.columns,
        index=numeric.index
    )

    infinite_counts.update(
        infinite_mask.sum().to_dict()
    )

    # Duplicates within each individual chunk only
    duplicate_rows_within_chunks += int(chunk.duplicated().sum())

print("=" * 55)
print("DATASET SUMMARY")
print("=" * 55)

print(f"Total records: {total_rows:,}")
print(f"Total features including label: {len(columns)}")

print("\nLABEL DISTRIBUTION")
for label, count in label_counts.most_common():
    percentage = count / total_rows * 100
    print(f"{label}: {count:,} ({percentage:.2f}%)")

print("\nMISSING VALUES")
missing_found = {
    col: count for col, count in missing_counts.items()
    if count > 0
}
print(missing_found if missing_found else "No missing values found.")

print("\nINFINITE VALUES")
infinite_found = {
    col: count for col, count in infinite_counts.items()
    if count > 0
}
print(infinite_found if infinite_found else "No infinite values found.")

print(
    "\nDuplicate rows within individual chunks:",
    f"{duplicate_rows_within_chunks:,}"
)
print("\nAnalysis complete.")
