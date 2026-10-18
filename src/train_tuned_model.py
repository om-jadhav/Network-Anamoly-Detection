
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

TRAIN_FILE = DATA_DIR / "cicids2017_train.csv"

MODEL_FILE = MODEL_DIR / "network_anomaly_model_tuned.joblib"
SCALER_FILE = MODEL_DIR / "network_anomaly_scaler_tuned.joblib"
FEATURES_FILE = MODEL_DIR / "network_anomaly_features.joblib"
REPORT_FILE = OUTPUT_DIR / "validation_threshold_results.txt"

CHUNK_SIZE = 25_000
EPOCHS = 3

# Original split used hash % 100 < 20 for test.
# Within the original training file:
# 20-39 = validation
# 40-99 = new training subset


# --------------------------------------------------
# 2. Read chunks and recreate the original hash split
# --------------------------------------------------

def read_chunks():
    return pd.read_csv(
        TRAIN_FILE,
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False,
    )


def get_masks(chunk, feature_columns):
    hashes = pd.util.hash_pandas_object(
        chunk[feature_columns],
        index=False,
    ).to_numpy(dtype="uint64")

    buckets = hashes % 100

    validation_mask = (buckets >= 20) & (buckets < 40)
    training_mask = buckets >= 40

    return training_mask, validation_mask


def numeric_features(chunk, feature_columns):
    return chunk[feature_columns].to_numpy(dtype=np.float64)


# --------------------------------------------------
# 3. Fit scaler using training records only
# --------------------------------------------------

def fit_scaler(feature_columns):
    scaler = StandardScaler()
    rows_seen = 0

    print("\nSTEP 1: Fitting scaler on training records")

    for chunk in read_chunks():
        train_mask, _ = get_masks(chunk, feature_columns)

        if not train_mask.any():
            continue

        X = numeric_features(
            chunk.loc[train_mask],
            feature_columns,
        )

        scaler.partial_fit(X)
        rows_seen += len(X)

        print(f"Scaler processed {rows_seen:,} rows")

    print(f"Scaler fitted on {rows_seen:,} training records.")
    return scaler


# --------------------------------------------------
# 4. Train a fresh model without validation records
# --------------------------------------------------

def train_model(scaler, feature_columns):
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        class_weight={0: 1.0, 1: 4.0},
        random_state=42,
        average=True,
        shuffle=True,
    )

    print("\nSTEP 2: Training the new model")

    for epoch in range(EPOCHS):
        rows_seen = 0
        first_batch = True

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for chunk in read_chunks():
            train_mask, _ = get_masks(chunk, feature_columns)

            if not train_mask.any():
                continue

            selected = chunk.loc[train_mask]
            X = numeric_features(selected, feature_columns)
            y = selected["Target"].astype("int8").to_numpy()

            X_scaled = scaler.transform(X)

            if first_batch:
                model.partial_fit(
                    X_scaled,
                    y,
                    classes=np.array([0, 1]),
                )
                first_batch = False
            else:
                model.partial_fit(X_scaled, y)

            rows_seen += len(selected)

            print(
                f"\rTraining records processed: {rows_seen:,}",
                end="",
                flush=True,
            )

        print(f"\nEpoch {epoch + 1} completed.")

    return model


# --------------------------------------------------
# 5. Collect predictions on held-out validation data
# --------------------------------------------------

def get_validation_predictions(model, scaler, feature_columns):
    actual_parts = []
    probability_parts = []
    rows_seen = 0

    print("\nSTEP 3: Predicting validation records")

    for chunk in read_chunks():
        _, validation_mask = get_masks(chunk, feature_columns)

        if not validation_mask.any():
            continue

        selected = chunk.loc[validation_mask]
        X = numeric_features(selected, feature_columns)
        y = selected["Target"].astype("int8").to_numpy()

        probabilities = model.predict_proba(
            scaler.transform(X)
        )[:, 1]

        actual_parts.append(y)
        probability_parts.append(probabilities)

        rows_seen += len(selected)
        print(f"Validation records processed: {rows_seen:,}")

    return (
        np.concatenate(actual_parts),
        np.concatenate(probability_parts),
    )


# --------------------------------------------------
# 6. Compare decision thresholds
# --------------------------------------------------

def evaluate_thresholds(y_true, probabilities):
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    report_lines = []

    report_lines.append("VALIDATION THRESHOLD COMPARISON")
    report_lines.append("=" * 78)
    report_lines.append(f"Validation records: {len(y_true):,}")
    report_lines.append(
        "\nThreshold | Precision | Recall | F1-score | False-positive rate"
    )
    report_lines.append("-" * 78)

    print("\nSTEP 4: Comparing decision thresholds")
    print("-" * 78)
    print(
        "Threshold | Precision | Recall | F1-score | False-positive rate"
    )
    print("-" * 78)

    for threshold in thresholds:
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

        false_positive_rate = fp / (fp + tn) if (fp + tn) else 0

        line = (
            f"{threshold:9.2f} | {precision:9.4f} | "
            f"{recall:6.4f} | {f1:8.4f} | "
            f"{false_positive_rate:18.4f}"
        )

        print(line)
        report_lines.append(line)

    REPORT_FILE.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print(f"\nResults saved to: {REPORT_FILE}")


# --------------------------------------------------
# 7. Run the pipeline
# --------------------------------------------------

def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    first_chunk = next(read_chunks())

    feature_columns = [
        column
        for column in first_chunk.columns
        if column not in ("Label", "Target")
    ]

    print("=" * 60)
    print("NETWORK ANOMALY DETECTION - VALIDATION EXPERIMENT")
    print("=" * 60)
    print(f"Features: {len(feature_columns)}")
    print(f"Chunk size: {CHUNK_SIZE:,}")
    print(f"Epochs: {EPOCHS}")
    print("Original baseline files will not be overwritten.")
    print("No additional dataset CSV files will be created.")

    scaler = fit_scaler(feature_columns)
    model = train_model(scaler, feature_columns)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(feature_columns, FEATURES_FILE)

    print("\nTuned-experiment model and scaler saved.")

    y_true, probabilities = get_validation_predictions(
        model,
        scaler,
        feature_columns,
    )

    evaluate_thresholds(y_true, probabilities)

    print("\nValidation experiment completed!")
    print("The test dataset was not used.")


if __name__ == "__main__":
    main()
