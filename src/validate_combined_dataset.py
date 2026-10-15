
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "cicids2017_combined.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "cicids2017_validated.csv"

CHUNK_SIZE = 100_000


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    total_rows = 0
    removed_negative_duration = 0
    removed_nonfinite = 0
    removed_both = 0
    removed_total = 0
    saved_rows = 0
    first_chunk = True

    for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
        chunk.columns = chunk.columns.str.strip()
        total_rows += len(chunk)

        feature_columns = [
            column for column in chunk.columns
            if column != "Label"
        ]

        numeric = chunk[feature_columns].apply(
            pd.to_numeric, errors="coerce"
        )

        # Detect missing, infinite, or non-numeric feature values.
        values = numeric.to_numpy(dtype=np.float64)
        nonfinite_mask = ~np.isfinite(values).all(axis=1)

        # A flow cannot have a negative duration.
        duration = pd.to_numeric(
            chunk["Flow Duration"], errors="coerce"
        )
        negative_duration_mask = duration < 0

        # Count each reason, including rows that meet both conditions.
        removed_negative_duration += int(
            negative_duration_mask.sum()
        )
        removed_nonfinite += int(nonfinite_mask.sum())
        removed_both += int(
            (negative_duration_mask & nonfinite_mask).sum()
        )

        # Remove the union of both conditions, not their sum.
        remove_mask = negative_duration_mask | nonfinite_mask
        cleaned = chunk.loc[~remove_mask]

        removed_total += int(remove_mask.sum())
        saved_rows += len(cleaned)

        cleaned.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False

        print(
            f"Checked {total_rows:,} rows | "
            f"Kept {saved_rows:,} rows"
        )

    print("\n" + "=" * 50)
    print("VALIDATION COMPLETE")
    print("=" * 50)
    print(f"Original rows: {total_rows:,}")
    print(f"Negative-duration rows: {removed_negative_duration:,}")
    print(f"Rows with non-finite features: {removed_nonfinite:,}")
    print(f"Rows meeting both conditions: {removed_both:,}")
    print(f"Total rows removed: {removed_total:,}")
    print(f"Validated rows remaining: {saved_rows:,}")
    print(f"Output: {OUTPUT_FILE}")
    print("\nOther negative values have been preserved.")
    print("Duplicate removal has not been performed.")


if __name__ == "__main__":
    main()
