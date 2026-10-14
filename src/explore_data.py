
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "Friday-DDos-cleaned.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv(DATA_FILE)

    print("\nDataset shape:", df.shape)

    # 1. Display the class distribution
    print("\nClass distribution:")
    print(df["Label"].value_counts())

    # 2. Display a few useful feature statistics
    feature_names = [
        "Destination Port",
        "Flow Duration",
        "Total Fwd Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
        "SYN Flag Count",
    ]

    print("\nSelected feature statistics:")
    print(df[feature_names].describe().round(2).to_string())

    # 3. Visualize the class distribution
    label_counts = df["Label"].value_counts()

    plt.figure(figsize=(8, 5))
    label_counts.plot(kind="bar")
    plt.title("Benign vs DDoS Traffic")
    plt.xlabel("Traffic Label")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=0)
    plt.tight_layout()

    chart_path = OUTPUT_DIR / "label_distribution.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print("\nChart saved to:", chart_path)

    # 4. Visualize flow duration by class
    plt.figure(figsize=(8, 5))

    for label in df["Label"].unique():
        values = df.loc[df["Label"] == label, "Flow Duration"]
        # Log scale helps display highly skewed durations.
        plt.hist(
            values.clip(lower=0),
            bins=50,
            alpha=0.5,
            label=label,
        )

    plt.xscale("symlog")
    plt.title("Flow Duration Distribution by Traffic Label")
    plt.xlabel("Flow Duration (microseconds, symlog scale)")
    plt.ylabel("Number of Records")
    plt.legend()
    plt.tight_layout()

    duration_path = OUTPUT_DIR / "flow_duration_distribution.png"
    plt.savefig(duration_path, dpi=150)
    plt.close()

    print("Chart saved to:", duration_path)
    print("\nExploration completed!")


if __name__ == "__main__":
    main()
