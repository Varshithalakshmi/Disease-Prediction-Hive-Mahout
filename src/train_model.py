
from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

TARGET = "Diabetes_binary"


def train_model():
    print("Loading training and testing data...")

    train_data = pd.read_csv(DATA_DIR / "train.csv")
    test_data = pd.read_csv(DATA_DIR / "test.csv")

    X_train = train_data.drop(columns=[TARGET])
    y_train = train_data[TARGET]

    X_test = test_data.drop(columns=[TARGET])
    y_test = test_data[TARGET]

    print("Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=60,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=2
    )

    model.fit(X_train, y_train)

    print("Making predictions...")
    predictions = model.predict(X_test)

    print("\nMODEL EVALUATION")
    print("-------------------------")

    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("Precision:", round(precision_score(y_test, predictions), 4))
    print("Recall:", round(recall_score(y_test, predictions), 4))
    print("F1 Score:", round(f1_score(y_test, predictions), 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, RESULTS_DIR / "random_forest_model.pkl")

    print("\nModel saved successfully!")
    print("Location:", RESULTS_DIR / "random_forest_model.pkl")


if __name__ == "__main__":
    train_model()
