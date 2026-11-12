from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Results from the completed model experiments.
data = [
    {
        "Model": "Mahout Logistic Regression",
        "Accuracy": 75.50,
        "Precision": 29.96,
        "Recall": 56.68,
        "F1 Score": 39.20
    },
    {
        "Model": "Random Forest",
        "Accuracy": 72.51,
        "Precision": 30.77,
        "Recall": 77.82,
        "F1 Score": 44.10
    },
    {
        "Model": "Federated Logistic Regression",
        "Accuracy": 66.32,
        "Precision": 26.97,
        "Recall": 82.97,
        "F1 Score": 40.70
    }
]

df = pd.DataFrame(data)

csv_path = RESULTS_DIR / "model_comparison.csv"
df.to_csv(csv_path, index=False)

print("\nMODEL PERFORMANCE COMPARISON")
print(df.to_string(index=False))

# Generate performance chart.
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

ax = df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(11, 6)
)

ax.set_title("Disease Prediction Model Performance")
ax.set_ylabel("Score (%)")
ax.set_xlabel("Model")
ax.set_ylim(0, 100)
ax.legend(title="Metrics")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()

chart_path = RESULTS_DIR / "model_comparison.png"
plt.savefig(chart_path, dpi=300)
plt.close()

print("\nComparison CSV saved:", csv_path)
print("Comparison chart saved:", chart_path)
