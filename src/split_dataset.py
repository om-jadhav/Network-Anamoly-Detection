
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "cicids2017_binary.csv"
)
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

TRAIN_FILE = OUTPUT_DIR / "cicids2017_train.csv"
TEST_FILE = OUTPUT_DIR / "cicids2017_test.csv"

CHUNK_SIZE = 100_000
TEST_PERCENT = 20


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    # Check available disk space before running:
    # the split creates two additional CSV files.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in (TRAIN_FILE, TEST_FILE):
        if path.exists():
            path.unlink()

    total_train = 0
    total_test = 0
    train_targets = {0: 0, 1: 0}
    test_targets = {0: 0, 1: 0}
    first_train = True
    first_test = True

    print("Creating feature-grouped train/test split...\n")

    # Read as strings so feature hashing is consistent
    # across chunks with different numeric-value ranges.
    for chunk in pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False,
    ):
        chunk.columns = chunk.columns.str.strip()

        if "Target" not in chunk.columns:
            raise ValueError("Target column is missing.")

        # Neither the original label nor the target is a feature.
        feature_columns = [
            column for column in chunk.columns
            if column not in ("Label", "Target")
        ]

        # Identical feature vectors get the same hash and split.
        hashes = pd.util.hash_pandas_object(
            chunk[feature_columns],
            index=False,
        ).to_numpy(dtype="uint64")

        is_test = (hashes % 100) < TEST_PERCENT

        train_chunk = chunk.loc[~is_test]
        test_chunk = chunk.loc[is_test]

        train_chunk.to_csv(
            TRAIN_FILE,
            mode="w" if first_train else "a",
            header=first_train,
            index=False,
        )
        test_chunk.to_csv(
            TEST_FILE,
            mode="w" if first_test else "a",
            header=first_test,
            index=False,
        )

        first_train = False
        first_test = False

        train_targets_chunk = train_chunk["Target"].astype(int)
        test_targets_chunk = test_chunk["Target"].astype(int)

        for target in (0, 1):
            train_targets[target] += int(
                (train_targets_chunk == target).sum()
            )
            test_targets[target] += int(
                (test_targets_chunk == target).sum()
            )

        total_train += len(train_chunk)
        total_test += len(test_chunk)

        print(
            f"Processed {total_train + total_test:,} rows | "
            f"Train: {total_train:,} | Test: {total_test:,}"
        )

    print("\n" + "=" * 50)
    print("TRAIN/TEST SPLIT COMPLETE")
    print("=" * 50)
    print(f"Training records: {total_train:,}")
    print(f"Testing records: {total_test:,}")
    print(f"Total records: {total_train + total_test:,}")

    print("\nTraining target distribution:")
    print(f"Normal (0): {train_targets[0]:,}")
    print(f"Attack (1): {train_targets[1]:,}")

    print("\nTesting target distribution:")
    print(f"Normal (0): {test_targets[0]:,}")
    print(f"Attack (1): {test_targets[1]:,}")

    print(f"\nTraining file: {TRAIN_FILE}")
    print(f"Testing file: {TEST_FILE}")
    print("\nIdentical feature vectors are kept in one split.")


if __name__ == "__main__":
    main()
