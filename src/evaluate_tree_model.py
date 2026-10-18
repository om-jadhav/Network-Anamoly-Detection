
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parent.parent

TEST_FILE = ROOT / "data" / "processed" / "cicids2017_test.csv"
MODEL_FILE = ROOT / "models" / "network_anomaly_tree_model.joblib"
FEATURES_FILE = ROOT / "models" / "network_anomaly_tree_features.joblib"
OUTPUT_FILE = ROOT / "outputs" / "tree_test_evaluation.txt"

CHUNK_SIZE = 25_000
THRESHOLD = 0.20


def main():
    print("Loading saved tree model...")

    model = joblib.load(MODEL_FILE)
    features = joblib.load(FEATURES_FILE)

    tn = fp = fn = tp = 0
    total = 0

    print("\nEvaluating on the held-out test dataset...")
    print(f"Decision threshold: {THRESHOLD:.2f}")

    for chunk in pd.read_csv(TEST_FILE, chunksize=CHUNK_SIZE):
        X = chunk[features].to_numpy(dtype=np.float32)
        y_true = chunk["Target"].to_numpy(dtype=np.int8)

        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= THRESHOLD).astype(np.int8)

        tn += int(np.sum((y_true == 0) & (predictions == 0)))
        fp += int(np.sum((y_true == 0) & (predictions == 1)))
        fn += int(np.sum((y_true == 1) & (predictions == 0)))
        tp += int(np.sum((y_true == 1) & (predictions == 1)))

        total += len(y_true)
        print(f"Evaluated {total:,} records...", end="\r")

    y_true_counts = np.array([tn + fp, fn + tp])
    predicted_counts = np.array([tn + fn, fp + tp])

    accuracy = (tn + tp) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    fpr = fp / (fp + tn) if fp + tn else 0.0

    report = classification_report(
        [0] * (tn + fn) + [1] * (fp + tp),
        # We calculate the actual report below from aggregate counts
        # using the confusion matrix instead.
        [0] * (tn + fn) + [1] * (fp + tp),
    ) if False else None

    # Class-specific precision, recall and F1 from the confusion matrix.
    normal_precision = tn / (tn + fn) if tn + fn else 0.0
    normal_recall = tn / (tn + fp) if tn + fp else 0.0
    normal_f1 = (
        2 * normal_precision * normal_recall
        / (normal_precision + normal_recall)
        if normal_precision + normal_recall
        else 0.0
    )

    lines = [
        "TREE MODEL - HELD-OUT TEST EVALUATION",
        f"Test records: {total:,}",
        f"Threshold: {THRESHOLD:.2f}",
        f"Accuracy: {accuracy:.4f}",
        f"Attack precision: {precision:.4f}",
        f"Attack recall: {recall:.4f}",
        f"Attack F1-score: {f1:.4f}",
        f"False-positive rate: {fpr:.4f}",
        "",
        "Confusion matrix (rows = actual, columns = predicted):",
        f"[[{tn}, {fp}],",
        f" [{fn}, {tp}]]",
        "",
        "Class-specific results:",
        "Class | Precision | Recall | F1-score | Support",
        (
            f"Normal | {normal_precision:.4f} | "
            f"{normal_recall:.4f} | {normal_f1:.4f} | {tn + fp}"
        ),
        (
            f"Attack | {precision:.4f} | {recall:.4f} | "
            f"{f1:.4f} | {fn + tp}"
        ),
    ]

    print("\n\n" + "\n".join(lines))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
