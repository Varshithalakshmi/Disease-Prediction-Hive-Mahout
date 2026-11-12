
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Project folders
PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Evaluation results from the common test dataset
comparison_data = {
    "Model": [
        "Random Forest",
        "Mahout Logistic Regression",
        "Federated Learning"
    ],
    "Accuracy": [
        0.7251,
        0.7724,
        0.6632
    ],
    "Precision": [
        0.3077,
        0.3143,
        0.2697
    ],
    "Recall": [
        0.7782,
        0.5364,
        0.8297
    ],
    "F1 Score": [
        0.4410,
        0.3964,
        0.4070
    ]
}


# Create comparison DataFrame
comparison_df = pd.DataFrame(comparison_data)


# Display results
print("\nCOMMON TEST SET MODEL COMPARISON")
print(comparison_df.to_string(index=False))


# Save CSV file
csv_path = RESULTS_DIR / "model_comparison_common_test.csv"
comparison_df.to_csv(csv_path, index=False)


# Create comparison chart
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

ax = comparison_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(11, 6),
    width=0.8
)

plt.title("Model Performance Comparison on Common Test Set")
plt.xlabel("Machine Learning Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15, ha="right")
plt.legend(title="Evaluation Metrics")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

# Save chart
chart_path = RESULTS_DIR / "model_comparison_common_test.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()


print("\nComparison files saved successfully:")
print(csv_path)
print(chart_path)
