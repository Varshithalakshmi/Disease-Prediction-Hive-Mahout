
from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score
)

PROJECT_DIR = Path(__file__).resolve().parents[1]

model = joblib.load(
    PROJECT_DIR / "results" / "random_forest_model.pkl"
)

test_data = pd.read_csv(
    PROJECT_DIR / "data" / "processed" / "test.csv"
)

X_test = test_data.drop(columns=["Diabetes_binary"])
y_test = test_data["Diabetes_binary"]

predictions = model.predict(X_test)

results = test_data.copy()
results["Actual"] = y_test
results["Predicted"] = predictions

# Convert age codes into readable groups
results["Age_Group"] = pd.cut(
    results["Age"],
    bins=[0, 3, 6, 9, 13],
    labels=["18-34", "35-49", "50-64", "65+"]
)

def evaluate_group(data, group_name):
    return {
        "Group": group_name,
        "Records": len(data),
        "Accuracy": round(
            accuracy_score(data["Actual"], data["Predicted"]), 4
        ),
        "Precision": round(
            precision_score(
                data["Actual"], data["Predicted"], zero_division=0
            ), 4
        ),
        "Recall": round(
            recall_score(
                data["Actual"], data["Predicted"], zero_division=0
            ), 4
        )
    }

# Audit by sex
sex_results = []

for value, group in results.groupby("Sex"):
    sex_name = "Female" if value == 0 else "Male"
    sex_results.append(evaluate_group(group, sex_name))

# Audit by age
age_results = []

for value, group in results.groupby("Age_Group", observed=True):
    age_results.append(evaluate_group(group, str(value)))

sex_df = pd.DataFrame(sex_results)
age_df = pd.DataFrame(age_results)

print("\nFAIRNESS AUDIT BY SEX")
print(sex_df.to_string(index=False))

print("\nFAIRNESS AUDIT BY AGE")
print(age_df.to_string(index=False))

output_dir = PROJECT_DIR / "results"
sex_df.to_csv(output_dir / "fairness_by_sex.csv", index=False)
age_df.to_csv(output_dir / "fairness_by_age.csv", index=False)

print("\nFairness audit files saved successfully!")
