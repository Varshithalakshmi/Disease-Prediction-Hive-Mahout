from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

TARGET = "Diabetes_binary"
HOSPITALS = 3
ROUNDS = 5
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.1
L2 = 0.001


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z))


def main():
    print("Loading datasets...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    X_train = train.drop(columns=[TARGET]).to_numpy(dtype=float)
    y_train = train[TARGET].to_numpy(dtype=float)

    X_test = test.drop(columns=[TARGET]).to_numpy(dtype=float)
    y_test = test[TARGET].to_numpy(dtype=float)

    # Simulate three hospitals with separate local datasets.
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X_train))
    hospital_indices = np.array_split(indices, HOSPITALS)

    local_data = [
        (X_train[idx], y_train[idx]) for idx in hospital_indices
    ]

    # Aggregate feature statistics without combining raw records.
    total = len(X_train)
    feature_sum = sum(X.sum(axis=0) for X, _ in local_data)
    feature_sq_sum = sum((X ** 2).sum(axis=0) for X, _ in local_data)

    mean = feature_sum / total
    variance = np.maximum(feature_sq_sum / total - mean ** 2, 1e-8)
    scale = np.sqrt(variance)

    local_data = [
        ((X - mean) / scale, y) for X, y in local_data
    ]
    X_test = (X_test - mean) / scale

    # Shared initial global model.
    global_weights = np.zeros(X_train.shape[1])
    global_bias = 0.0

    print("\nStarting Federated Learning...")
    print("Number of simulated hospitals:", HOSPITALS)

    for round_number in range(ROUNDS):
        hospital_weights = []
        hospital_biases = []
        hospital_sizes = []

        for hospital_id, (X, y) in enumerate(local_data, start=1):
            weights = global_weights.copy()
            bias = global_bias

            # Balanced class weights for each hospital.
            positives = max(np.sum(y == 1), 1)
            negatives = max(np.sum(y == 0), 1)

            sample_weights = np.where(
                y == 1,
                len(y) / (2 * positives),
                len(y) / (2 * negatives)
            )

            # Local training; raw hospital records stay local.
            for _ in range(LOCAL_EPOCHS):
                probabilities = sigmoid(X @ weights + bias)
                errors = (probabilities - y) * sample_weights

                gradient = (X.T @ errors) / len(y) + L2 * weights
                bias_gradient = np.mean(errors)

                weights -= LEARNING_RATE * gradient
                bias -= LEARNING_RATE * bias_gradient

            hospital_weights.append(weights)
            hospital_biases.append(bias)
            hospital_sizes.append(len(y))

        # FedAvg: weighted aggregation of local model parameters.
        total_records = sum(hospital_sizes)

        global_weights = sum(
            w * size
            for w, size in zip(hospital_weights, hospital_sizes)
        ) / total_records

        global_bias = sum(
            b * size
            for b, size in zip(hospital_biases, hospital_sizes)
        ) / total_records

        print(f"Federated round {round_number + 1}/{ROUNDS} completed.")

    print("\nEvaluating federated model...")
    predictions = (
        sigmoid(X_test @ global_weights + global_bias) >= 0.5
    ).astype(int)

    print("\nFEDERATED MODEL EVALUATION")
    print("--------------------------")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
    print("Recall:", round(recall_score(y_test, predictions, zero_division=0), 4))
    print("F1 Score:", round(f1_score(y_test, predictions, zero_division=0), 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        RESULTS_DIR / "federated_model.npz",
        weights=global_weights,
        bias=global_bias,
        mean=mean,
        scale=scale
    )

    print("\nFederated model saved successfully!")
    print("Location:", RESULTS_DIR / "federated_model.npz")


if __name__ == "__main__":
    main()
