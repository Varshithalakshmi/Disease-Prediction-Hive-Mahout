# Disease Prediction Using Hive and Apache Mahout

## 1. Project Overview

This project develops a data-driven diabetes prediction system using medical and lifestyle information. It combines Apache Hive for large-scale data analysis, Apache Mahout for machine learning, Random Forest for model comparison, Explainable AI for feature interpretation, fairness auditing, and simulated federated learning across three hospitals.

**Purpose:** Educational and research use only. This project is not intended for clinical diagnosis or medical decision-making.

## 2. Objectives

- Store and analyze diabetes data using Apache Hive.
- Build a diabetes prediction model using Apache Mahout.
- Compare Mahout, Random Forest, and Federated Logistic Regression.
- Identify important prediction features using feature importance.
- Evaluate model performance across sex and age groups.
- Simulate federated learning across three hospitals.
- Study the trade-off between accuracy, precision, recall, and F1 score.

## 3. Technologies Used

- Python
- Java
- Apache Hive
- Apache Hadoop
- Apache Mahout
- Scikit-learn
- Pandas and NumPy
- Matplotlib
- Maven

## 4. Dataset

The project uses a diabetes health indicators dataset containing 253,680 valid records and 23 columns.

The target variable is `Diabetes_binary`:
- 0: Non-diabetic
- 1: Diabetic

The dataset contains health and lifestyle features such as BMI, high blood pressure, high cholesterol, physical activity, age category, and general health.

There are 21 input features after excluding the ID and target columns.

## 5. System Architecture

1. Dataset: Medical and lifestyle data is stored in CSV format.
2. Data processing: Python prepares the training and testing datasets.
3. Hive analysis: Apache Hive performs data validation, aggregation, and statistical analysis.
4. Model development: Apache Mahout and Random Forest build prediction models.
5. Explainable AI: Feature importance identifies influential model inputs.
6. Fairness audit: Performance is evaluated across demographic groups.
7. Federated learning: Three simulated hospitals train local models and share model updates for aggregation.
8. Evaluation: The models are compared using standard classification metrics.

## 6. Machine Learning Models

### Apache Mahout Logistic Regression
A Java-based Online Logistic Regression model trained with class oversampling and validation-based threshold selection.

### Random Forest
A Python-based Random Forest classifier trained with class-balanced weights.

### Federated Logistic Regression
A simulated federated model involving three hospital data partitions. Local model parameters are combined using Federated Averaging (FedAvg).

## 7. Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Mahout Logistic Regression | 75.50% | 29.96% | 56.68% | 39.20% |
| Random Forest | 72.51% | 30.77% | 77.82% | 44.10% |
| Federated Logistic Regression | 66.32% | 26.97% | 82.97% | 40.70% |

### Results Interpretation

- Mahout achieved the highest accuracy among the three experiments.
- Random Forest achieved the highest F1 score.
- Federated Logistic Regression achieved the highest recall.
- The federated model detected a larger proportion of positive cases, but had lower precision and accuracy.

**Important evaluation limitation:** The Mahout experiment used a 70/15/15 train-validation-test split, while the Python experiments used an 80/20 train-test split. These results are indicative and are not a strictly controlled comparison on an identical test set.

## 8. Explainable AI

Random Forest feature importance was used to identify the most influential features.

Top features included:
1. High blood pressure
2. General health
3. BMI
4. High cholesterol
5. Age category

These values describe how the trained model uses features. They do not establish medical causation or explain an individual prediction.

## 9. Fairness Analysis

The model was evaluated across sex and age groups.

The audit showed differences in accuracy, precision, and recall across groups. These differences highlight potential fairness concerns that require further investigation. They do not by themselves prove discrimination.

## 10. Federated Learning

The federated learning experiment simulates three hospitals.

- Each hospital trains a local logistic regression model.
- Model parameters are shared for aggregation.
- Federated Averaging combines the local model updates.
- The experiment saves the resulting model as `results/federated_model.npz`.

This is a simulation, not a production privacy-preserving deployment. It does not implement secure aggregation or differential privacy.

## 11. Project Structure

```text
Disease-Prediction-Hive-Mahout/
├── data/
│   ├── raw/
│   └── processed/
├── hive/
│   └── sql/
│       └── diabetes_analysis.sql
├── mahout/
│   ├── backup/
│   └── pom.xml
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── explore_data.py
│   ├── explain_model.py
│   ├── fairness_audit.py
│   ├── federated_learning.py
│   └── compare_models.py
├── results/
│   ├── fairness_by_age.csv
│   ├── fairness_by_sex.csv
│   ├── feature_importance.csv
│   ├── federated_model.npz
│   ├── hive_analysis_output.txt
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   └── random_forest_model.pkl
└── README.md
```

## 12. Running the Python Modules

Install the required packages:

```bash
python -m pip install pandas scikit-learn joblib matplotlib
```

Run the modules from the project root:

```bash
python src/preprocessing.py
python src/train_model.py
python src/explain_model.py
python src/fairness_audit.py
python src/federated_learning.py
python src/compare_models.py
```

The raw dataset must be available at `data/raw/diabetes.csv`.

## 13. Limitations

- The dataset contains an imbalanced target distribution.
- The model comparison uses different train-test splits.
- Federated learning is simulated in one environment.
- Secure aggregation and differential privacy are not implemented.
- Feature importance does not establish causation.
- The models are research prototypes and are not clinically validated.

## 14. Future Enhancements

- Evaluate every model using the same train-test split.
- Improve prediction performance and calibration.
- Add individual-level explanation methods such as SHAP.
- Investigate fairness mitigation techniques.
- Implement secure aggregation and differential privacy.
- Develop a user-friendly prediction interface.
- Validate models using independent datasets.

## 15. Conclusion

This project demonstrates how big-data tools and machine learning can be combined for diabetes risk prediction research. Apache Hive supports data analysis, Apache Mahout and Random Forest provide prediction models, Explainable AI supports feature interpretation, fairness auditing examines group-level performance, and federated learning demonstrates collaborative model training across simulated hospitals.

The results show that different models have different strengths. Further testing on a common dataset split, privacy improvements, fairness evaluation, and independent validation are necessary before considering real-world use.
