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
OUTPUT_FILE = OUTPUT_DIR / "threshold_tradeoff_results.txt"

CHUNK_SIZE = 25_000

# Thresholds to compare
THRESHOLDS = [0.20, 0.15, 0.10, 0.075, 0.05, 0.02, 0.01]


def main():
    print("Loading saved tree model and feature list...")

    model = joblib.load(MODEL_FILE)
    features = joblib.load(FEATURES_FILE)

    # Store aggregate counts for each threshold.
    results = {
        threshold: {
            "bot_detected": 0,
            "bot_missed": 0,
            "benign_detected_as_attack": 0,
            "benign_correct": 0,
            "total_attacks_detected": 0,
            "total_attacks_missed": 0,
        }
        for threshold in THRESHOLDS
    }

    total_records = 0
    total_bot_records = 0
    total_benign_records = 0
    total_attack_records = 0

    print("\nAnalyzing thresholds on the existing test dataset...")

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

        # Preserve original attack labels.
        labels = chunk["Label"].str.strip().to_numpy()

        # Binary ground truth: BENIGN=0, attacks=1.
        targets = pd.to_numeric(
            chunk["Target"], errors="raise"
        ).to_numpy(dtype=np.int8)

        # Convert model features in the correct order.
        X = chunk[features].apply(
            pd.to_numeric,
            errors="raise",
        ).to_numpy(dtype=np.float32)

        # Predict probabilities using the saved model.
        probabilities = model.predict_proba(X)[:, 1]

        bot_mask = labels == "Bot"
        benign_mask = targets == 0
        attack_mask = targets == 1

        total_records += len(chunk)
        total_bot_records += int(bot_mask.sum())
        total_benign_records += int(benign_mask.sum())
        total_attack_records += int(attack_mask.sum())

        for threshold in THRESHOLDS:
            predicted_attack = probabilities >= threshold

            counts = results[threshold]

            counts["bot_detected"] += int(
                np.sum(predicted_attack & bot_mask)
            )
            counts["bot_missed"] += int(
                np.sum((~predicted_attack) & bot_mask)
            )

            counts["benign_detected_as_attack"] += int(
                np.sum(predicted_attack & benign_mask)
            )
            counts["benign_correct"] += int(
                np.sum((~predicted_attack) & benign_mask)
            )

            counts["total_attacks_detected"] += int(
                np.sum(predicted_attack & attack_mask)
            )
            counts["total_attacks_missed"] += int(
                np.sum((~predicted_attack) & attack_mask)
            )

        print(
            f"Processed chunk {chunk_number} "
            f"({total_records:,} records)"
        )

    # Build results table.
    rows = []

    for threshold in THRESHOLDS:
        counts = results[threshold]

        bot_recall = (
            counts["bot_detected"] / total_bot_records
            if total_bot_records else 0.0
        )

        false_positive_rate = (
            counts["benign_detected_as_attack"] / total_benign_records
            if total_benign_records else 0.0
        )

        overall_attack_recall = (
            counts["total_attacks_detected"] / total_attack_records
            if total_attack_records else 0.0
        )

        rows.append({
            "Threshold": threshold,
            "Bot Detected": counts["bot_detected"],
            "Bot Missed": counts["bot_missed"],
            "Bot Recall (%)": round(bot_recall * 100, 2),
            "False Positives": counts["benign_detected_as_attack"],
            "False Positive Rate (%)": round(
                false_positive_rate * 100, 4
            ),
            "Overall Attack Recall (%)": round(
                overall_attack_recall * 100, 2
            ),
        })

    results_df = pd.DataFrame(rows)

    report_lines = [
        "NETWORK ANOMALY DETECTION",
        "THRESHOLD TRADE-OFF ANALYSIS",
        "=" * 100,
        f"Test dataset: {TEST_FILE.name}",
        f"Model: {MODEL_FILE.name}",
        f"Total records: {total_records:,}",
        f"Total BENIGN records: {total_benign_records:,}",
        f"Total attack records: {total_attack_records:,}",
        f"Total Bot records: {total_bot_records:,}",
        "",
        "RESULTS BY THRESHOLD",
        "-" * 100,
        results_df.to_string(index=False),
        "",
        "INTERPRETATION",
        "-" * 100,
        "Lower thresholds classify more records as attacks.",
        "This can improve attack recall but may increase false positives.",
        "False-positive rate = BENIGN records flagged as attacks / all BENIGN records.",
        "",
        "Note: These are exploratory results on the existing test set.",
        "Do not use this test-set analysis as an independent final evaluation",
        "or to claim generalization to unseen network traffic.",
        "",
        "The saved model and test dataset were not modified.",
    ]

    report = "\n".join(report_lines)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\nReport saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()