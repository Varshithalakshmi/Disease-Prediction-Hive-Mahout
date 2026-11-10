
from pathlib import Path
import pandas as pd
import joblib

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_FILE = PROJECT_DIR / "results" / "random_forest_model.pkl"
TEST_FILE = PROJECT_DIR / "data" / "processed" / "test.csv"

model = joblib.load(MODEL_FILE)
test_data = pd.read_csv(TEST_FILE)

X_test = test_data.drop(columns=["Diabetes_binary"])

importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 10 IMPORTANT FEATURES")
print(importance.head(10).to_string(index=False))

importance.to_csv(
    PROJECT_DIR / "results" / "feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully!")
