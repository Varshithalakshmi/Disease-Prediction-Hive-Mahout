
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_FILE = PROJECT_DIR / "results" / "random_forest_model.pkl"
TEST_FILE = PROJECT_DIR / "data" / "processed" / "test.csv"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# Load model and test data
print("Loading model and test data...")

model = joblib.load(MODEL_FILE)
test_data = pd.read_csv(TEST_FILE)

X_test = test_data.drop(columns=["Diabetes_binary"])

# Use a reproducible sample to keep SHAP computation manageable
X_sample = X_test.sample(
    n=min(1000, len(X_test)),
    random_state=42
)

print("Test sample size:", len(X_sample))

# ------------------------------------------------
# 1. Existing Random Forest feature importance
# ------------------------------------------------

importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    RESULTS_DIR / "feature_importance.csv",
    index=False
)

print("\nTOP 10 RANDOM FOREST FEATURES")
print(importance.head(10).to_string(index=False))

# ------------------------------------------------
# 2. SHAP global explainability
# ------------------------------------------------

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(model)
raw_shap_values = explainer.shap_values(X_sample)

# Select the positive class (diabetes indicator)
if isinstance(raw_shap_values, list):
    positive_shap_values = raw_shap_values[1]

else:
    positive_shap_values = np.asarray(raw_shap_values)

    if positive_shap_values.ndim == 3:
        if positive_shap_values.shape[-1] == 2:
            positive_shap_values = positive_shap_values[:, :, 1]
        elif positive_shap_values.shape[0] == 2:
            positive_shap_values = positive_shap_values[1, :, :]
        else:
            raise ValueError(
                "Unexpected SHAP output shape: "
                + str(positive_shap_values.shape)
            )

# Global SHAP importance
shap_importance = pd.DataFrame({
    "Feature": X_sample.columns,
    "Mean_Absolute_SHAP": np.abs(
        positive_shap_values
    ).mean(axis=0)
})

shap_importance = shap_importance.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)

shap_importance.to_csv(
    RESULTS_DIR / "shap_feature_importance.csv",
    index=False
)

print("\nTOP 10 SHAP FEATURES")
print(shap_importance.head(10).to_string(index=False))

# SHAP summary plot
plt.figure()

shap.summary_plot(
    positive_shap_values,
    X_sample,
    show=False,
    max_display=15
)

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "shap_summary.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# ------------------------------------------------
# 3. Individual prediction explanation
# ------------------------------------------------

print("\nGenerating individual explanation...")

sample_index = 0

local_values = positive_shap_values[sample_index]

local_explanation = pd.DataFrame({
    "Feature": X_sample.columns,
    "Input_Value": X_sample.iloc[sample_index].values,
    "SHAP_Contribution": local_values
})

local_explanation["Absolute_Contribution"] = (
    local_explanation["SHAP_Contribution"].abs()
)

local_explanation = local_explanation.sort_values(
    by="Absolute_Contribution",
    ascending=False
)

local_explanation.to_csv(
    RESULTS_DIR / "shap_local_explanation.csv",
    index=False
)

# Local explanation chart
plot_data = local_explanation.head(15).sort_values(
    by="SHAP_Contribution"
)

plt.figure(figsize=(9, 6))

plt.barh(
    plot_data["Feature"],
    plot_data["SHAP_Contribution"]
)

plt.axvline(0, linewidth=1)
plt.xlabel("SHAP contribution to positive-class model output")
plt.ylabel("Health indicators")
plt.title("Individual Diabetes Prediction Explanation")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "shap_local_explanation.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\nSHAP EXPLANATIONS GENERATED SUCCESSFULLY!")

print("\nSaved files:")
print("1. results/feature_importance.csv")
print("2. results/shap_feature_importance.csv")
print("3. results/shap_summary.png")
print("4. results/shap_local_explanation.csv")
print("5. results/shap_local_explanation.png")
