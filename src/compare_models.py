
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier


# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = ROOT / "data" / "processed" / "cicids2017_train.csv"
MODEL_FILE = ROOT / "models" / "network_anomaly_model_tuned.joblib"
SCALER_FILE = ROOT / "models" / "network_anomaly_scaler_tuned.joblib"
FEATURES_FILE = ROOT / "models" / "network_anomaly_features.joblib"

OUTPUT_FILE = ROOT / "outputs" / "model_comparison.txt"

CHUNK_SIZE = 25_000
SAMPLE_RATE = 0.06
THRESHOLDS = [0.10, 0.12, 0.20]

RANDOM_SEED = 42


# --------------------------------------------------
# 2. Load the existing tuned SGD model
# --------------------------------------------------

print("\nLoading the existing tuned SGD model...")

sgd_model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
features = joblib.load(FEATURES_FILE)

print("SGD model loaded successfully.")


# --------------------------------------------------
# 3. Sample training rows and train a tree model
# --------------------------------------------------

print("\nReading training data and selecting a sample...")

rng = np.random.default_rng(RANDOM_SEED)
X_parts = []
y_parts = []

start_time = time.time()

for chunk in pd.read_csv(TRAIN_FILE, chunksize=CHUNK_SIZE):
    # Recreate the existing hash-based data split.
    hashes = pd.util.hash_pandas_object(
        chunk[features], index=False
    ).to_numpy(dtype=np.uint64)

    buckets = hashes % 100

    # Only use the training partition.
    # Buckets 20-39 are reserved for validation.
    train_mask = buckets >= 40

    # Randomly sample across the whole dataset,
    # rather than taking only the first rows.
    sample_mask = train_mask & (
        rng.random(len(chunk)) < SAMPLE_RATE
    )

    if sample_mask.any():
        X_parts.append(
            chunk.loc[sample_mask, features].to_numpy(
                dtype=np.float32
            )
        )

        y_parts.append(
            chunk.loc[sample_mask, "Target"].to_numpy(
                dtype=np.int8
            )
        )

if not X_parts:
    raise RuntimeError(
        "No training samples found. Check the dataset path "
        "and feature columns."
    )

X_train = np.concatenate(X_parts, axis=0)
y_train = np.concatenate(y_parts)

del X_parts, y_parts

normal_count = np.sum(y_train == 0)
attack_count = np.sum(y_train == 1)

print(f"Training sample size: {len(y_train):,}")
print(f"Normal samples: {normal_count:,}")
print(f"Attack samples: {attack_count:,}")

if normal_count == 0 or attack_count == 0:
    raise RuntimeError(
        "The sample must contain both normal and attack records."
    )

# Balance the total training weight of the two classes.
sample_weights = np.where(
    y_train == 1,
    len(y_train) / (2 * attack_count),
    len(y_train) / (2 * normal_count),
)

print("\nTraining HistGradientBoostingClassifier...")

tree_model = HistGradientBoostingClassifier(
    max_iter=100,
    max_leaf_nodes=31,
    learning_rate=0.1,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=RANDOM_SEED,
)

tree_model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
)

# Save the trained model for later evaluation.
model_dir = ROOT / "models"
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(
    tree_model,
    model_dir / "network_anomaly_tree_model.joblib",
)

joblib.dump(
    features,
    model_dir / "network_anomaly_tree_features.joblib",
)

print("Tree model trained and saved successfully.")

training_seconds = time.time() - start_time

del X_train, y_train, sample_weights


# --------------------------------------------------
# 4. Evaluate both models on the SAME validation set
# --------------------------------------------------

print("\nEvaluating on the validation partition...")

# Store TN, FP, FN, TP for each model and threshold.
counts = {
    model_name: {
        threshold: [0, 0, 0, 0]
        for threshold in THRESHOLDS
    }
    for model_name in ["SGD", "HistGradientBoosting"]
}

validation_records = 0

for chunk in pd.read_csv(TRAIN_FILE, chunksize=CHUNK_SIZE):
    hashes = pd.util.hash_pandas_object(
        chunk[features], index=False
    ).to_numpy(dtype=np.uint64)

    buckets = hashes % 100

    # Only buckets 20-39 belong to validation.
    validation_mask = (buckets >= 20) & (buckets < 40)

    if not validation_mask.any():
        continue

    X_val = chunk.loc[
        validation_mask, features
    ].to_numpy(dtype=np.float32)

    y_true = chunk.loc[
        validation_mask, "Target"
    ].to_numpy(dtype=np.int8)

    # SGD requires the scaler fitted on its training partition.
    X_val_scaled = scaler.transform(X_val)

    sgd_probabilities = sgd_model.predict_proba(
        X_val_scaled
    )[:, 1]

    tree_probabilities = tree_model.predict_proba(
        X_val
    )[:, 1]

    probabilities = {
        "SGD": sgd_probabilities,
        "HistGradientBoosting": tree_probabilities,
    }

    validation_records += len(y_true)

    for model_name, probs in probabilities.items():
        for threshold in THRESHOLDS:
            predictions = (probs >= threshold).astype(np.int8)

            tn = np.sum((y_true == 0) & (predictions == 0))
            fp = np.sum((y_true == 0) & (predictions == 1))
            fn = np.sum((y_true == 1) & (predictions == 0))
            tp = np.sum((y_true == 1) & (predictions == 1))

            current = counts[model_name][threshold]
            current[0] += int(tn)
            current[1] += int(fp)
            current[2] += int(fn)
            current[3] += int(tp)

    print(f"Validated {validation_records:,} records...", end="\r")

print(f"\nTotal validation records: {validation_records:,}")


# --------------------------------------------------
# 5. Print and save comparison results
# --------------------------------------------------

lines = [
    "NETWORK ANOMALY DETECTION - MODEL COMPARISON",
    f"Validation records: {validation_records:,}",
    f"Tree training sample: {len(counts['SGD']) and 'sampled training partition'}",
    f"Tree training time: {training_seconds:.1f} seconds",
    "",
    "Model | Threshold | Precision | Recall | F1-score | False-positive rate",
]

print("\n" + lines[-1])
print("-" * 82)

for model_name in ["SGD", "HistGradientBoosting"]:
    for threshold in THRESHOLDS:
        tn, fp, fn, tp = counts[model_name][threshold]

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        fpr = fp / (fp + tn) if fp + tn else 0.0

        line = (
            f"{model_name} | {threshold:.2f} | "
            f"{precision:.4f} | {recall:.4f} | "
            f"{f1:.4f} | {fpr:.4f}"
        )

        print(line)
        lines.append(line)

        if model_name == "HistGradientBoosting":
            pass

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

print(f"\nResults saved to: {OUTPUT_FILE}")
