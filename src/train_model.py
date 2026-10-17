
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

TRAIN_FILE = DATA_DIR / "cicids2017_train.csv"
TEST_FILE = DATA_DIR / "cicids2017_test.csv"

MODEL_FILE = MODEL_DIR / "network_anomaly_model.joblib"
SCALER_FILE = MODEL_DIR / "network_anomaly_scaler.joblib"
REPORT_FILE = OUTPUT_DIR / "model_evaluation.txt"

CHUNK_SIZE = 50_000
EPOCHS = 3


# --------------------------------------------------
# 2. Read the dataset in manageable chunks
# --------------------------------------------------

def read_chunks(file_path):
    return pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
    )


def get_features(chunk):
    # Label is the original attack category.
    # Target is the answer we want the model to predict.
    # Neither should be used as an input feature.
    return [
        column
        for column in chunk.columns
        if column not in ("Label", "Target")
    ]


# --------------------------------------------------
# 3. Fit the scaler using training data only
# --------------------------------------------------

def fit_scaler(feature_columns):
    scaler = StandardScaler()
    total_rows = 0

    print("\nSTEP 1: Learning feature scaling values")
    print("-" * 50)

    for chunk in read_chunks(TRAIN_FILE):
        X = chunk[feature_columns]

        scaler.partial_fit(X)
        total_rows += len(chunk)

        print(f"Scaler processed {total_rows:,} rows")

    print("Feature scaling completed.")
    return scaler


# --------------------------------------------------
# 4. Train the model incrementally
# --------------------------------------------------

def train_model(scaler, feature_columns):
    # Attack records are less common than normal records.
    # Give attack classification errors more weight.
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        class_weight={0: 1.0, 1: 4.0},
        random_state=42,
        average=True,
        shuffle=True,
    )

    print("\nSTEP 2: Training the classifier")
    print("-" * 50)

    for epoch in range(EPOCHS):
        total_rows = 0

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for chunk in read_chunks(TRAIN_FILE):
            X = chunk[feature_columns]
            y = chunk["Target"].astype("int8").to_numpy()

            X_scaled = scaler.transform(X)

            # partial_fit learns from one batch at a time.
            if epoch == 0 and total_rows == 0:
                model.partial_fit(
                    X_scaled,
                    y,
                    classes=np.array([0, 1]),
                )
            else:
                model.partial_fit(X_scaled, y)

            total_rows += len(chunk)

            print(
                f"\rTraining progress: {total_rows:,} rows",
                end="",
                flush=True,
            )

        print(f"\nEpoch {epoch + 1} completed.")

    return model


# --------------------------------------------------
# 5. Evaluate on unseen test data
# --------------------------------------------------

def evaluate_model(model, scaler, feature_columns):
    actual_values = []
    predicted_values = []
    total_rows = 0

    print("\nSTEP 3: Evaluating on test data")
    print("-" * 50)

    for chunk in read_chunks(TEST_FILE):
        X = chunk[feature_columns]
        y = chunk["Target"].astype("int8").to_numpy()

        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)

        actual_values.append(y)
        predicted_values.append(predictions)

        total_rows += len(chunk)
        print(f"Evaluated {total_rows:,} test rows")

    y_true = np.concatenate(actual_values)
    y_pred = np.concatenate(predicted_values)

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    results = (
        "NETWORK ANOMALY DETECTION - BASELINE RESULTS\n"
        + "=" * 50
        + f"\nTest records: {len(y_true):,}"
        + f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)"
        + "\n\nClassification report:\n"
        + report
        + "\nConfusion matrix (rows=actual, columns=predicted):\n"
        + str(matrix)
        + "\n\nMatrix order: Normal (0), Attack (1)\n"
    )

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(results)

    REPORT_FILE.write_text(results, encoding="utf-8")
    print(f"\nEvaluation report saved to: {REPORT_FILE}")


# --------------------------------------------------
# 6. Run the complete training pipeline
# --------------------------------------------------

def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Testing dataset not found: {TEST_FILE}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover feature columns from the training data.
    first_chunk = next(read_chunks(TRAIN_FILE))
    feature_columns = get_features(first_chunk)

    if "Target" not in first_chunk.columns:
        raise ValueError("The Target column is missing.")

    print("=" * 50)
    print("AI-BASED NETWORK ANOMALY DETECTION")
    print("=" * 50)
    print(f"Features: {len(feature_columns)}")
    print(f"Chunk size: {CHUNK_SIZE:,}")
    print(f"Training epochs: {EPOCHS}")
    print("Normal = 0 | Attack = 1")

    # Pass 1: calculate scaling statistics.
    scaler = fit_scaler(feature_columns)

    # Passes 2-4: train for three epochs.
    model = train_model(scaler, feature_columns)

    # Save both artifacts for later predictions.
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    print(f"\nModel saved to: {MODEL_FILE}")
    print(f"Scaler saved to: {SCALER_FILE}")

    # Evaluate only after training is complete.
    evaluate_model(model, scaler, feature_columns)

    print("\nTraining and evaluation finished!")


if __name__ == "__main__":
    main()
