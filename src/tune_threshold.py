
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "cicids2017_train.csv"
)
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_FILE = MODEL_DIR / "network_anomaly_model_tuned.joblib"
SCALER_FILE = MODEL_DIR / "network_anomaly_scaler_tuned.joblib"
FEATURES_FILE = MODEL_DIR / "network_anomaly_features.joblib"
REPORT_FILE = OUTPUT_DIR / "fine_threshold_results.txt"

CHUNK_SIZE = 25_000

# Explore the promising region between 0.10 and 0.20.
THRESHOLDS = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]


def main():
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    features = joblib.load(FEATURES_FILE)

    actual_parts = []
    probability_parts = []

    print("Collecting validation predictions...")

    for chunk in pd.read_csv(
        DATA_FILE,
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False,
    ):
        hashes = pd.util.hash_pandas_object(
            chunk[features],
            index=False,
        ).to_numpy(dtype="uint64")

        buckets = hashes % 100
        validation_mask = (buckets >= 20) & (buckets < 40)

        if not validation_mask.any():
            continue

        selected = chunk.loc[validation_mask]

        X = selected[features].to_numpy(dtype=np.float64)
        y = selected["Target"].astype("int8").to_numpy()

        probabilities = model.predict_proba(
            scaler.transform(X)
        )[:, 1]

        actual_parts.append(y)
        probability_parts.append(probabilities)

    y_true = np.concatenate(actual_parts)
    probabilities = np.concatenate(probability_parts)

    lines = [
        f"Validation records: {len(y_true):,}",
        "",
        "Threshold | Precision | Recall | F1 | False-positive rate",
        "-" * 65,
    ]

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype("int8")

        precision = precision_score(
            y_true, predictions, zero_division=0
        )
        recall = recall_score(
            y_true, predictions, zero_division=0
        )
        f1 = f1_score(
            y_true, predictions, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()

        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        lines.append(
            f"{threshold:9.2f} | {precision:9.4f} | "
            f"{recall:6.4f} | {f1:5.4f} | {fpr:18.4f}"
        )

        print(lines[-1])

    REPORT_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved results to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
