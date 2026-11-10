
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_DIR / "data" / "raw" / "diabetes.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

TARGET = "Diabetes_binary"


def preprocess_data():

    print("Loading original dataset...")

    df = pd.read_csv(RAW_FILE)

    # Remove record ID because it is not a medical feature
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Check target column
    if TARGET not in df.columns:
        raise ValueError("Target column not found!")

    # Check missing values
    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values.")

    # Separate input features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print("Total records:", len(df))
    print("Input features:", X.shape[1])

    # Split data while preserving class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Reattach target for saving
    train_data = X_train.copy()
    train_data[TARGET] = y_train

    test_data = X_test.copy()
    test_data[TARGET] = y_test

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False
    )

    test_data.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False
    )

    print("\nPreprocessing completed!")

    print("Training records:", len(train_data))
    print("Testing records:", len(test_data))

    print("\nTraining class distribution:")
    print(y_train.value_counts(normalize=True).round(4))

    print("\nTesting class distribution:")
    print(y_test.value_counts(normalize=True).round(4))

    print("\nFiles saved in:", PROCESSED_DIR)


if __name__ == "__main__":
    preprocess_data()
