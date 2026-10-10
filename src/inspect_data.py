
from pathlib import Path
import pandas as pd

# Locate the dataset folder relative to the project root
DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "MachineLearningCSV"
)

# Find all CSV files
csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {DATA_DIR.resolve()}"
    )

print(f"Found {len(csv_files)} CSV files\n")

for file in csv_files:
    size_mb = file.stat().st_size / (1024 * 1024)
    print(f"{file.name}: {size_mb:.2f} MB")

# Select the Friday DDoS dataset
target = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
file_path = DATA_DIR / target

if not file_path.exists():
    raise FileNotFoundError(
        f"Selected dataset not found: {file_path}"
    )

# Read only 5 rows for initial inspection
df = pd.read_csv(file_path, nrows=5, encoding="utf-8-sig")

print("\nSelected file:", file_path.name)
print("\nColumns:")
print(df.columns.tolist())

print("\nSample records:")
print(df.to_string(index=False))
