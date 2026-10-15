
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "cicids2017_combined.csv"

CHUNK_SIZE = 100_000

CHECK_COLUMNS = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Min",
    "Flow IAT Mean",
    "Flow IAT Max",
    "Fwd IAT Min",
    "Fwd Header Length",
    "Fwd Header Length.1",
    "Bwd Header Length",
    "min_seg_size_forward",
]

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    negative_by_column = {}
    negative_labels = {}
    invalid_flow_duration_labels = {}
    samples = []
    total_rows = 0

    for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
        chunk.columns = chunk.columns.str.strip()
        total_rows += len(chunk)

        for column in CHECK_COLUMNS:
            if column not in chunk.columns:
                continue

            values = pd.to_numeric(chunk[column], errors="coerce")
            mask = values < 0
            count = int(mask.sum())

            if count:
                negative_by_column[column] = (
                    negative_by_column.get(column, 0) + count
                )

                label_counts = chunk.loc[mask, "Label"].value_counts()
                for label, label_count in label_counts.items():
                    key = (column, str(label))
                    negative_labels[key] = (
                        negative_labels.get(key, 0) + int(label_count)
                    )

        # Focus on records with negative duration
        duration = pd.to_numeric(
            chunk["Flow Duration"], errors="coerce"
        )
        bad_duration = duration < 0

        if bad_duration.any():
            label_counts = chunk.loc[
                bad_duration, "Label"
            ].value_counts()

            for label, count in label_counts.items():
                label = str(label)
                invalid_flow_duration_labels[label] = (
                    invalid_flow_duration_labels.get(label, 0)
                    + int(count)
                )

            if len(samples) < 10:
                sample = chunk.loc[bad_duration].head(
                    10 - len(samples)
                )
                columns_to_show = [
                    c for c in [
                        "Label",
                        "Flow Duration",
                        "Flow Bytes/s",
                        "Flow Packets/s",
                        "Flow IAT Min",
                        "Init_Win_bytes_forward",
                        "Init_Win_bytes_backward",
                    ]
                    if c in sample.columns
                ]
                samples.extend(
                    sample[columns_to_show].to_dict("records")
                )

    print("\nTOTAL ROWS INSPECTED:", f"{total_rows:,}")

    print("\nNEGATIVE VALUES BY FEATURE")
    for column, count in sorted(
        negative_by_column.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{column}: {count:,}")

    print("\nNEGATIVE VALUES BY FEATURE AND LABEL")
    for (column, label), count in sorted(
        negative_labels.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{column} | {label}: {count:,}")

    print("\nNEGATIVE FLOW DURATION BY LABEL")
    if invalid_flow_duration_labels:
        for label, count in sorted(
            invalid_flow_duration_labels.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"{label}: {count:,}")
    else:
        print("None found.")

    print("\nEXAMPLE RECORDS WITH NEGATIVE FLOW DURATION")
    if samples:
        print(pd.DataFrame(samples).to_string(index=False))
    else:
        print("No examples to display.")

    print("\nInspection complete. No data was modified.")

if __name__ == "__main__":
    main()
