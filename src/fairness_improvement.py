
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# PROJECT PATHS AND SETTINGS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

TARGET = "Diabetes_binary"
TARGET_RECALL = 0.70


# ==========================================
# AGE GROUP CREATION
# ==========================================

def add_age_group(data):

    result = data.copy()

    result["Age_Group"] = pd.cut(
        result["Age"],
        bins=[0, 3, 6, 9, 13],
        labels=["18-34", "35-49", "50-64", "65+"]
    )

    return result


# ==========================================
# METRICS
# ==========================================

def evaluate_group(actual, predicted):

    return {
        "Records": len(actual),

        "Accuracy": round(
            accuracy_score(actual, predicted), 4
        ),

        "Precision": round(
            precision_score(
                actual, predicted, zero_division=0
            ), 4
        ),

        "Recall": round(
            recall_score(
                actual, predicted, zero_division=0
            ), 4
        ),

        "F1": round(
            f1_score(
                actual, predicted, zero_division=0
            ), 4
        ),

        "Positive_Rate": round(
            np.mean(predicted), 4
        )
    }


# ==========================================
# THRESHOLD CALIBRATION
# ==========================================

def find_threshold(actual, probabilities):

    best_threshold = 0.50
    best_difference = float("inf")

    for threshold in np.linspace(0.01, 0.99, 99):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        recall = recall_score(
            actual,
            predictions,
            zero_division=0
        )

        difference = abs(recall - TARGET_RECALL)

        if difference < best_difference:

            best_difference = difference
            best_threshold = threshold

        elif (
            difference == best_difference
            and threshold > best_threshold
        ):

            best_threshold = threshold

    return float(best_threshold)


# ==========================================
# MAIN EXPERIMENT
# ==========================================

def run_experiment():

    print("Loading dataset...")

    train_data = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    test_data = pd.read_csv(
        DATA_DIR / "test.csv"
    )

    X = train_data.drop(columns=[TARGET])
    y = train_data[TARGET]

    X_test = test_data.drop(columns=[TARGET])
    y_test = test_data[TARGET]

    # Split training data into model-training
    # and threshold-calibration subsets.

    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    print("Training experiment model...")

    model = RandomForestClassifier(
        n_estimators=60,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=2
    )

    model.fit(X_train, y_train)

    positive_index = list(model.classes_).index(1)

    # --------------------------------------
    # VALIDATION PROBABILITIES
    # --------------------------------------

    validation_probabilities = (
        model.predict_proba(X_validation)[:, positive_index]
    )

    validation_data = X_validation.copy()
    validation_data["Actual"] = y_validation.to_numpy()
    validation_data["Probability"] = validation_probabilities

    validation_data = add_age_group(validation_data)

    print("\nFinding age-specific thresholds...")

    thresholds = {}

    for age_group in [
        "18-34",
        "35-49",
        "50-64",
        "65+"
    ]:

        group = validation_data[
            validation_data["Age_Group"] == age_group
        ]

        threshold = find_threshold(
            group["Actual"],
            group["Probability"]
        )

        thresholds[age_group] = threshold

        print(
            age_group,
            "threshold:",
            round(threshold, 3)
        )

    # --------------------------------------
    # TEST SET PREDICTIONS
    # --------------------------------------

    test_probabilities = (
        model.predict_proba(X_test)[:, positive_index]
    )

    baseline_predictions = (
        test_probabilities >= 0.50
    ).astype(int)

    test_results = X_test.copy()

    test_results["Actual"] = y_test.to_numpy()
    test_results["Probability"] = test_probabilities

    test_results = add_age_group(test_results)

    adjusted_predictions = []

    for _, row in test_results.iterrows():

        age_group = str(row["Age_Group"])

        threshold = thresholds[age_group]

        prediction = int(
            row["Probability"] >= threshold
        )

        adjusted_predictions.append(prediction)

    adjusted_predictions = np.array(
        adjusted_predictions
    )

    # --------------------------------------
    # EVALUATE OVERALL AND GROUP METRICS
    # --------------------------------------

    rows = []

    overall_baseline = evaluate_group(
        y_test,
        baseline_predictions
    )

    overall_baseline.update({
        "Model": "Baseline",
        "Group_Type": "Overall",
        "Group": "All"
    })

    rows.append(overall_baseline)

    overall_adjusted = evaluate_group(
        y_test,
        adjusted_predictions
    )

    overall_adjusted.update({
        "Model": "Age-Adjusted",
        "Group_Type": "Overall",
        "Group": "All"
    })

    rows.append(overall_adjusted)

    evaluation_data = test_results.copy()

    evaluation_data["Baseline_Prediction"] = (
        baseline_predictions
    )

    evaluation_data["Adjusted_Prediction"] = (
        adjusted_predictions
    )

    # Age group evaluation

    for age_group in [
        "18-34",
        "35-49",
        "50-64",
        "65+"
    ]:

        group = evaluation_data[
            evaluation_data["Age_Group"] == age_group
        ]

        for model_name, prediction_column in [
            ("Baseline", "Baseline_Prediction"),
            ("Age-Adjusted", "Adjusted_Prediction")
        ]:

            metrics = evaluate_group(
                group["Actual"],
                group[prediction_column]
            )

            metrics.update({
                "Model": model_name,
                "Group_Type": "Age",
                "Group": age_group
            })

            rows.append(metrics)

    # Sex group evaluation

    for sex_value, group in evaluation_data.groupby("Sex"):

        sex_name = (
            "Female" if sex_value == 0 else "Male"
        )

        for model_name, prediction_column in [
            ("Baseline", "Baseline_Prediction"),
            ("Age-Adjusted", "Adjusted_Prediction")
        ]:

            metrics = evaluate_group(
                group["Actual"],
                group[prediction_column]
            )

            metrics.update({
                "Model": model_name,
                "Group_Type": "Sex",
                "Group": sex_name
            })

            rows.append(metrics)

    results = pd.DataFrame(rows)

    # --------------------------------------
    # SAVE RESULTS
    # --------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        RESULTS_DIR / "fairness_improvement_comparison.csv",
        index=False
    )

    threshold_results = pd.DataFrame(
        [
            {
                "Age_Group": group,
                "Threshold": round(threshold, 4)
            }
            for group, threshold in thresholds.items()
        ]
    )

    threshold_results.to_csv(
        RESULTS_DIR / "age_group_thresholds.csv",
        index=False
    )

    # Save the separate experiment model.
    # The existing Random Forest model is not overwritten.

    joblib.dump(
        model,
        RESULTS_DIR / "fairness_experiment_model.pkl"
    )

    # --------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------

    print("\nFAIRNESS IMPROVEMENT RESULTS")
    print("--------------------------------")

    print(
        results.to_string(index=False)
    )

    print("\nAge-specific thresholds:")
    print(
        threshold_results.to_string(index=False)
    )

    print("\nFiles saved:")
    print("results/fairness_improvement_comparison.csv")
    print("results/age_group_thresholds.csv")
    print("results/fairness_experiment_model.pkl")

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    run_experiment()
