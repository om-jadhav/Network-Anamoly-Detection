
from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "cicids2017_validated.csv"
DB_FILE = PROJECT_DIR / "data" / "processed" / "duplicate_check_temp.db"

CHUNK_SIZE = 100_000


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    if DB_FILE.exists():
        DB_FILE.unlink()

    total_rows = 0

    try:
        with sqlite3.connect(DB_FILE) as connection:
            connection.execute("""
                CREATE TABLE exact_rows (
                    row_hash TEXT PRIMARY KEY,
                    occurrences INTEGER NOT NULL
                )
            """)

            connection.execute("""
                CREATE TABLE feature_rows (
                    feature_hash TEXT PRIMARY KEY,
                    occurrences INTEGER NOT NULL,
                    first_label TEXT,
                    label_conflict INTEGER NOT NULL DEFAULT 0
                )
            """)

            for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
                chunk.columns = chunk.columns.str.strip()
                total_rows += len(chunk)

                if "Label" not in chunk.columns:
                    raise ValueError("Label column is missing.")

                feature_columns = [
                    col for col in chunk.columns if col != "Label"
                ]

                # Hash all columns to find exact duplicate records.
                exact_hashes = pd.util.hash_pandas_object(
                    chunk, index=False
                ).to_numpy(dtype="uint64")

                exact_records = [
                    (f"{int(value):016x}",)
                    for value in exact_hashes
                ]

                connection.executemany("""
                    INSERT INTO exact_rows (row_hash, occurrences)
                    VALUES (?, 1)
                    ON CONFLICT(row_hash)
                    DO UPDATE SET occurrences = occurrences + 1
                """, exact_records)

                # Hash only features to find identical inputs
                # even if their labels differ.
                feature_hashes = pd.util.hash_pandas_object(
                    chunk[feature_columns], index=False
                ).to_numpy(dtype="uint64")

                feature_records = [
                    (f"{int(value):016x}", str(label))
                    for value, label in zip(
                        feature_hashes, chunk["Label"]
                    )
                ]

                connection.executemany("""
                    INSERT INTO feature_rows (
                        feature_hash, occurrences, first_label,
                        label_conflict
                    )
                    VALUES (?, 1, ?, 0)
                    ON CONFLICT(feature_hash)
                    DO UPDATE SET
                        label_conflict = CASE
                            WHEN feature_rows.first_label
                                 != excluded.first_label
                            THEN 1
                            ELSE feature_rows.label_conflict
                        END,
                        occurrences = feature_rows.occurrences + 1
                """, feature_records)

                connection.commit()
                print(f"Checked {total_rows:,} rows")

            exact_duplicate_rows = connection.execute("""
                SELECT COALESCE(SUM(occurrences - 1), 0)
                FROM exact_rows
            """).fetchone()[0]

            feature_duplicate_rows = connection.execute("""
                SELECT COALESCE(SUM(occurrences - 1), 0)
                FROM feature_rows
            """).fetchone()[0]

            conflicting_groups = connection.execute("""
                SELECT COUNT(*)
                FROM feature_rows
                WHERE label_conflict = 1
            """).fetchone()[0]

            conflicting_extra_rows = connection.execute("""
                SELECT COALESCE(SUM(occurrences - 1), 0)
                FROM feature_rows
                WHERE label_conflict = 1
            """).fetchone()[0]

            print("\n" + "=" * 50)
            print("DUPLICATE CHECK RESULTS")
            print("=" * 50)
            print(f"Rows inspected: {total_rows:,}")
            print(
                "Exact duplicate occurrences beyond first copies: "
                f"{exact_duplicate_rows:,}"
            )
            print(
                "Feature duplicate occurrences beyond first copies: "
                f"{feature_duplicate_rows:,}"
            )
            print(
                "Feature groups with conflicting labels: "
                f"{conflicting_groups:,}"
            )
            print(
                "Extra occurrences in conflicting feature groups: "
                f"{conflicting_extra_rows:,}"
            )
            print("\nThe validated dataset was not modified.")

    finally:
        if DB_FILE.exists():
            DB_FILE.unlink()


if __name__ == "__main__":
    main()
