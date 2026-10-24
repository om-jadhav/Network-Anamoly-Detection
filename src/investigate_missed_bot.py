from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = ROOT_DIR / "data" / "processed" / "cicids2017_test.csv"
MODEL_FILE = ROOT_DIR / "models" / "network_anomaly_tree_model.joblib"
FEATURES_FILE = ROOT_DIR / "models" / "network_anomaly_tree_features.joblib"

OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "missed_bot_investigation.txt"

CHUNK_SIZE = 25_000
THRESHOLD = 0.20


def main():
    print("Loading saved tree model and feature list...")

    model = joblib.load(MODEL_FILE)
    features = joblib.load(FEATURES_FILE)

    missed_bot_count = 0
    detected_bot_count = 0

    missed_probabilities = []
    detected_probabilities = []

    missed_examples = []
    detected_examples = []

    print("\nInvestigating Bot predictions in the existing test dataset...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            TEST_FILE,
            chunksize=CHUNK_SIZE,
            dtype=str,
            keep_default_na=False,
        ),
        start=1,
    ):
        chunk.columns = chunk.columns.str.strip()

        # Keep the original attack labels.
        labels = chunk["Label"].str.strip()

        # Identify Bot records only.
        bot_mask = labels.eq("Bot").to_numpy()

        if not bot_mask.any():
            print(f"Processed chunk {chunk_number}")
            continue

        # Convert the model features to numeric values.
        X_bot = chunk.loc[bot_mask, features].apply(
            pd.to_numeric,
            errors="raise",
        ).to_numpy(dtype=np.float32)

        # Use the saved model without retraining.
        probabilities = model.predict_proba(X_bot)[:, 1]

        # Bot records are attacks, so predictions below the threshold
        # are false negatives.
        missed_mask = probabilities < THRESHOLD
        detected_mask = ~missed_mask

        missed_probs = probabilities[missed_mask]
        detected_probs = probabilities[detected_mask]

        missed_probabilities.extend(missed_probs.tolist())
        detected_probabilities.extend(detected_probs.tolist())

        missed_bot_count += len(missed_probs)
        detected_bot_count += len(detected_probs)

        # Retain a few examples for investigation.
        for probability in missed_probs[: max(0, 10 - len(missed_examples))]:
            missed_examples.append(float(probability))

        for probability in detected_probs[: max(0, 10 - len(detected_examples))]:
            detected_examples.append(float(probability))

        print(
            f"Processed chunk {chunk_number} | "
            f"Bot records found: {int(bot_mask.sum())}"
        )

    total_bot_count = missed_bot_count + detected_bot_count

    if total_bot_count == 0:
        print("No Bot records were found. Check the original Label values.")
        return

    missed_array = np.asarray(missed_probabilities)
    detected_array = np.asarray(detected_probabilities)

    report_lines = [
        "MISSED BOT ATTACK INVESTIGATION",
        "=" * 60,
        f"Test dataset: {TEST_FILE.name}",
        f"Model: {MODEL_FILE.name}",
        f"Decision threshold: {THRESHOLD:.2f}",
        f"Total Bot records: {total_bot_count}",
        f"Bot records detected: {detected_bot_count}",
        f"Bot records missed: {missed_bot_count}",
        (
            f"Bot detection recall: "
            f"{detected_bot_count / total_bot_count:.4f} "
            f"({detected_bot_count / total_bot_count * 100:.2f}%)"
        ),
        "",
        "MISSED BOT PROBABILITY STATISTICS",
        "-" * 60,
    ]

    if missed_array.size:
        report_lines.extend([
            f"Minimum: {missed_array.min():.6f}",
            f"Maximum: {missed_array.max():.6f}",
            f"Mean: {missed_array.mean():.6f}",
            f"Median: {np.median(missed_array):.6f}",
            f"90th percentile: {np.percentile(missed_array, 90):.6f}",
            "",
            "Sample missed probabilities:",
            ", ".join(f"{p:.6f}" for p in missed_examples),
        ])
    else:
        report_lines.append("No missed Bot records.")

    report_lines.extend([
        "",
        "DETECTED BOT PROBABILITY STATISTICS",
        "-" * 60,
    ])

    if detected_array.size:
        report_lines.extend([
            f"Minimum: {detected_array.min():.6f}",
            f"Maximum: {detected_array.max():.6f}",
            f"Mean: {detected_array.mean():.6f}",
            f"Median: {np.median(detected_array):.6f}",
            "",
            "Sample detected probabilities:",
            ", ".join(f"{p:.6f}" for p in detected_examples),
        ])

    report_lines.extend([
        "",
        "=" * 60,
        "Investigation completed.",
        "The saved model and test dataset were not modified.",
    ])

    report = "\n".join(report_lines)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\nReport saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()