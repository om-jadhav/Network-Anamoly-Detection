
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = ROOT_DIR / "data" / "processed" / "cicids2017_test.csv"
MODEL_FILE = ROOT_DIR / "models" / "network_anomaly_tree_model.joblib"
FEATURES_FILE = ROOT_DIR / "models" / "network_anomaly_tree_features.joblib"
OUTPUT_FILE = ROOT_DIR / "outputs" / "attack_type_evaluation.txt"

CHUNK_SIZE = 25_000
THRESHOLD = 0.20


def main():
    print("Loading saved tree model and feature list...")

    model = joblib.load(MODEL_FILE)
    features = joblib.load(FEATURES_FILE)

    print(f"Model: {MODEL_FILE.name}")
    print(f"Number of features: {len(features)}")
    print(f"Decision threshold: {THRESHOLD:.2f}")
    print("\nEvaluating the existing test dataset by original attack label...")

    # Store counts for each original CIC-IDS2017 label.
    label_counts = {}

    total_records = 0

    # Read the existing test file without creating another dataset.
    for chunk_number, chunk in enumerate(
        pd.read_csv(
            TEST_FILE,
            chunksize=CHUNK_SIZE,
            dtype=str,
            keep_default_na=False
        ),
        start=1
    ):
        chunk.columns = chunk.columns.str.strip()

        required_columns = {"Label", "Target"}
        missing_columns = required_columns - set(chunk.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        missing_features = [
            feature for feature in features
            if feature not in chunk.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing model features: {missing_features}"
            )

        # Preserve original labels for per-attack evaluation.
        labels = chunk["Label"].str.strip()
        targets = pd.to_numeric(
            chunk["Target"], errors="raise"
        ).to_numpy(dtype=np.int8)

        # Convert features to numeric values in the saved model's order.
        X = chunk[features].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(dtype=np.float32)

        # Use the saved model only; do not retrain it.
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= THRESHOLD).astype(np.int8)

        # Accumulate results for each original Label.
        for label in labels.unique():
            mask = (labels.to_numpy() == label)

            actual_targets = targets[mask]
            predicted_targets = predictions[mask]

            if label not in label_counts:
                label_counts[label] = {
                    "records": 0,
                    "actual_attacks": 0,
                    "detected_attacks": 0,
                    "missed_attacks": 0,
                    "false_positives": 0,
                    "correctly_normal": 0,
                }

            counts = label_counts[label]
            counts["records"] += int(mask.sum())

            attack_mask = actual_targets == 1
            normal_mask = actual_targets == 0

            counts["actual_attacks"] += int(attack_mask.sum())
            counts["detected_attacks"] += int(
                np.sum((predicted_targets == 1) & attack_mask)
            )
            counts["missed_attacks"] += int(
                np.sum((predicted_targets == 0) & attack_mask)
            )
            counts["false_positives"] += int(
                np.sum((predicted_targets == 1) & normal_mask)
            )
            counts["correctly_normal"] += int(
                np.sum((predicted_targets == 0) & normal_mask)
            )

        total_records += len(chunk)

        print(
            f"Processed chunk {chunk_number} "
            f"({total_records:,} records so far)"
        )

    # Build the report.
    report_lines = [
        "NETWORK ANOMALY DETECTION",
        "PER-ATTACK-TYPE EVALUATION",
        "=" * 65,
        f"Test dataset: {TEST_FILE.name}",
        f"Model: {MODEL_FILE.name}",
        f"Decision threshold: {THRESHOLD:.2f}",
        f"Total test records: {total_records:,}",
        "",
        "RESULTS BY ORIGINAL LABEL",
        "-" * 65,
    ]

    # Show BENIGN first, followed by attack labels alphabetically.
    ordered_labels = sorted(
        label_counts,
        key=lambda name: (
            name.upper() != "BENIGN",
            name.lower()
        )
    )

    for label in ordered_labels:
        counts = label_counts[label]

        report_lines.append(f"\nLabel: {label}")
        report_lines.append(f"  Total records: {counts['records']:,}")

        if label.upper() == "BENIGN":
            normal_total = counts["records"]
            false_positives = counts["false_positives"]
            correctly_normal = counts["correctly_normal"]

            false_positive_rate = (
                false_positives / normal_total
                if normal_total else 0.0
            )

            report_lines.append(
                f"  Correctly classified as normal: "
                f"{correctly_normal:,}"
            )
            report_lines.append(
                f"  Incorrectly flagged as attack: "
                f"{false_positives:,}"
            )
            report_lines.append(
                f"  False-positive rate: "
                f"{false_positive_rate:.4f} "
                f"({false_positive_rate * 100:.2f}%)"
            )
        else:
            actual_attacks = counts["actual_attacks"]
            detected = counts["detected_attacks"]
            missed = counts["missed_attacks"]

            recall = (
                detected / actual_attacks
                if actual_attacks else 0.0
            )

            report_lines.append(
                f"  Attack records: {actual_attacks:,}"
            )
            report_lines.append(
                f"  Detected attacks: {detected:,}"
            )
            report_lines.append(
                f"  Missed attacks: {missed:,}"
            )
            report_lines.append(
                f"  Detection recall: {recall:.4f} "
                f"({recall * 100:.2f}%)"
            )

    report_lines.extend([
        "",
        "=" * 65,
        "Evaluation completed. The saved model and test dataset "
        "were not modified.",
    ])

    report = "\n".join(report_lines)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\nReport saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
