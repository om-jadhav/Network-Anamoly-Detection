
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "cicids2017_validated.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "cicids2017_binary.csv"
)

CHUNK_SIZE = 100_000


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    total_rows = 0
    benign_rows = 0
    attack_rows = 0
    first_chunk = True

    print("Preparing binary classification dataset...\n")

    for chunk in pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
    ):
        chunk.columns = chunk.columns.str.strip()

        if "Label" not in chunk.columns:
            raise ValueError("Label column is missing.")

        # Create a binary target:
        # 0 = normal traffic
        # 1 = attack traffic
        labels = chunk["Label"].astype("string").str.strip()

        if labels.isna().any():
            raise ValueError("Missing labels detected.")

        chunk["Target"] = (labels != "BENIGN").astype("int8")

        benign_rows += int((chunk["Target"] == 0).sum())
        attack_rows += int((chunk["Target"] == 1).sum())
        total_rows += len(chunk)

        chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        first_chunk = False

        print(f"Processed {total_rows:,} rows")

    print("\n" + "=" * 45)
    print("BINARY DATASET PREPARATION COMPLETE")
    print("=" * 45)
    print(f"Total records: {total_rows:,}")
    print(f"Normal traffic (0): {benign_rows:,}")
    print(f"Attack traffic (1): {attack_rows:,}")
    print(f"Output file: {OUTPUT_FILE}")
    print("\nOriginal Label column has been preserved.")


if __name__ == "__main__":
    main()
