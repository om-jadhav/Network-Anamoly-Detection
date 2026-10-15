
from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "cicids2017_validated.csv"
DB_FILE = PROJECT_DIR / "data" / "processed" / "label_conflicts_temp.db"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "label_conflict_records.csv"

CHUNK_SIZE = 100_000


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    if DB_FILE.exists():
        DB_FILE.unlink()

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    try:
        with sqlite3.connect(DB_FILE) as connection:
            connection.execute("""
                CREATE TABLE feature_labels (
                    feature_hash TEXT PRIMARY KEY,
                    first_label TEXT,
                    has_conflict INTEGER DEFAULT 0
                )
            """)

            # Pass 1: identify feature hashes associated with
            # different labels.
            for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
                chunk.columns = chunk.columns.str.strip()
                features = chunk.drop(columns=["Label"])

                hashes = pd.util.hash_pandas_object(
                    features, index=False
                ).to_numpy(dtype="uint64")

                records = [
                    (f"{int(h):016x}", str(label))
                    for h, label in zip(hashes, chunk["Label"])
                ]

                connection.executemany("""
                    INSERT INTO feature_labels (
                        feature_hash, first_label, has_conflict
                    )
                    VALUES (?, ?, 0)
                    ON CONFLICT(feature_hash)
                    DO UPDATE SET has_conflict = CASE
                        WHEN feature_labels.first_label
                             != excluded.first_label
                        THEN 1
                        ELSE feature_labels.has_conflict
                    END
                """, records)

                connection.commit()

            conflict_count = connection.execute("""
                SELECT COUNT(*)
                FROM feature_labels
                WHERE has_conflict = 1
            """).fetchone()[0]

            print(f"Conflicting feature groups found: {conflict_count:,}")

            connection.execute("""
                CREATE TABLE conflicts (
                    feature_hash TEXT PRIMARY KEY
                )
            """)

            connection.execute("""
                INSERT INTO conflicts (feature_hash)
                SELECT feature_hash
                FROM feature_labels
                WHERE has_conflict = 1
            """)
            connection.commit()

            # Pass 2: export all records in those groups.
            first_chunk = True
            exported = 0

            for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
                chunk.columns = chunk.columns.str.strip()
                features = chunk.drop(columns=["Label"])

                hashes = pd.util.hash_pandas_object(
                    features, index=False
                ).to_numpy(dtype="uint64")

                hash_strings = [f"{int(h):016x}" for h in hashes]

                # SQLite parameter limits vary, so query in small batches.
                matching_hashes = set()
                for start in range(0, len(hash_strings), 500):
                    batch = hash_strings[start:start + 500]
                    placeholders = ",".join("?" for _ in batch)
                    query = f"""
                        SELECT feature_hash FROM conflicts
                        WHERE feature_hash IN ({placeholders})
                    """
                    matching_hashes.update(
                        row[0] for row in connection.execute(query, batch)
                    )

                mask = pd.Series(hash_strings).isin(matching_hashes)
                matches = chunk.loc[mask]

                if not matches.empty:
                    matches.to_csv(
                        OUTPUT_FILE,
                        mode="w" if first_chunk else "a",
                        header=first_chunk,
                        index=False,
                    )
                    first_chunk = False
                    exported += len(matches)

                print(f"Scanned {min(exported, 0):,} conflict rows exported so far", end="\r")

        print("\nInvestigation complete.")
        print(f"Conflicting feature groups: {conflict_count:,}")
        print(f"Records exported: {exported:,}")
        print(f"Review file: {OUTPUT_FILE}")
        print("The validated dataset was not modified.")

    finally:
        # Close the SQLite connection before removing its file.
        if DB_FILE.exists():
            try:
                DB_FILE.unlink()
            except PermissionError:
                print(f"Could not remove temporary database: {DB_FILE}")


if __name__ == "__main__":
    main()
