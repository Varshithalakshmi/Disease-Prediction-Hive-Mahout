
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "raw" / "diabetes.csv"


def explore_dataset():

    print("Loading diabetes dataset...")

    df = pd.read_csv(DATA_PATH)

    # Remove ID because it is not a medical feature
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    print("\n========== DATASET INFORMATION ==========")

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate records:")
    print(df.duplicated().sum())

    print("\nFirst five records:")
    print(df.head())

    # Target distribution
    print("\n========== TARGET DISTRIBUTION ==========")

    print(df["Diabetes_binary"].value_counts())

    print("\nTarget percentages:")
    print(
        df["Diabetes_binary"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\n========== EXPLORATION COMPLETED ==========")


if __name__ == "__main__":
    explore_dataset()
