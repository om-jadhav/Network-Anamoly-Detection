
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "Friday-DDos-cleaned.csv"
)


def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv(DATA_FILE)

    # Identify numeric columns
    numeric_df = df.select_dtypes(include="number")

    # Count negative values in every numeric feature
    negative_counts = (numeric_df < 0).sum()
    negative_counts = negative_counts[negative_counts > 0]
    negative_counts = negative_counts.sort_values(ascending=False)

    print("\n" + "=" * 55)
    print("NEGATIVE VALUE SUMMARY")
    print("=" * 55)

    if negative_counts.empty:
        print("No negative values found.")
    else:
        print(negative_counts.to_string())

    # Investigate important features by traffic label
    features = [
        "Flow Duration",
        "Flow Bytes/s",
        "Flow Packets/s",
    ]

    print("\n" + "=" * 55)
    print("NEGATIVE VALUES BY LABEL")
    print("=" * 55)

    for feature in features:
        if feature not in df.columns:
            continue

        print(f"\nFeature: {feature}")

        negative_rows = df[df[feature] < 0]

        if negative_rows.empty:
            print("No negative values.")
            continue

        print(
            negative_rows.groupby("Label")[feature]
            .agg(["count", "min", "max"])
            .to_string()
        )

        print("\nExample rows:")
        print(
            negative_rows[["Label", feature]]
            .head(5)
            .to_string(index=False)
        )

    print("\nInvestigation completed.")


if __name__ == "__main__":
    main()
