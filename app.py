
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT / "results"
DATA = PROJECT / "data" / "processed"
MODEL_PATH = RESULTS / "random_forest_model.pkl"


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)


# ==========================================
# GLOBAL DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1 {
    color: #12345a;
}

h2, h3 {
    color: #176b87;
}

[data-testid="stSidebar"] {
    background-color: #eaf2f8;
}

div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}

.hero {
    background: linear-gradient(120deg, #102a43, #176b87);
    padding: 45px 35px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 38px;
}

.hero p {
    color: #e5f4fa;
    font-size: 17px;
}

.info-card {
    background: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #e1e8ef;
    min-height: 145px;
    margin-bottom: 15px;
}

.info-card h3 {
    color: #176b87;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL AND DATA
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA / "test.csv")


def load_result(filename):
    path = RESULTS / filename

    if path.exists():
        return pd.read_csv(path)

    return None


if not MODEL_PATH.exists():
    st.error("Random Forest model not found.")
    st.stop()

try:
    model = load_model()

except Exception as error:
    st.error("Unable to load the model.")
    st.exception(error)
    st.stop()


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🩺 Diabetes AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Home",
        "Diabetes Prediction",
        "Dataset Analysis",
        "Model Comparison",
        "Explainable AI",
        "Fairness Analysis",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Research Platform

Machine Learning  
Big Data Analytics  
Healthcare Research
""")

st.sidebar.caption("Academic Research Project")


# ==========================================
# HOME PAGE
# ==========================================

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>🩺 Diabetes Prediction System</h1>
        <p>
        Intelligent Healthcare Analysis Using Machine Learning
        and Big Data Technologies
        </p>
        <p>
        Explore health indicators, analyse diabetes-related data,
        compare machine learning models and understand model behaviour.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Project Overview")

    st.write("""
    Our research project combines big-data processing and
    machine learning to analyse diabetes-related health indicators.

    It integrates Apache Hive, Apache Mahout and Random Forest,
    along with Explainable AI, fairness analysis and simulated
    federated learning.
    """)

    st.markdown("## Project Highlights")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Dataset</h3>
            <h2>253,680</h2>
            <p>Health survey records</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <h3>🧠 Features</h3>
            <h2>21</h2>
            <p>Model input features</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="info-card">
            <h3>🤖 Main Model</h3>
            <h2>Random Forest</h2>
            <p>Machine learning model</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Explore the Platform")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("""
        <div class="info-card">
            <h3>🔍 Diabetes Prediction</h3>
            <p>Enter health indicators and view the model prediction.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>📈 Model Comparison</h3>
            <p>Compare Random Forest, Mahout Logistic Regression
            and Federated Learning results.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>⚖️ Fairness Analysis</h3>
            <p>Examine model performance across demographic groups.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="info-card">
            <h3>📊 Dataset Analysis</h3>
            <p>Explore dataset statistics and class distribution.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>🧠 Explainable AI</h3>
            <p>Discover which features contribute most to model predictions.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>🏥 Federated Learning</h3>
            <p>Explore the simulated multi-hospital learning approach.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.info("""
    Research disclaimer: This project is intended for academic
    and educational purposes. Predictions are not medical diagnoses.
    """)


# ==========================================
# DIABETES PREDICTION
# ==========================================

elif page == "Diabetes Prediction":

    st.title("🔍 Diabetes Prediction")

    st.write(
        "Enter numerical values for the health indicators below."
    )

    st.warning(
        "The current input range is 1 to 100 as requested. "
        "However, the original dataset uses different ranges and "
        "encodings for several features. Predictions may therefore "
        "be unreliable for arbitrary inputs."
    )

    with st.form("prediction_form"):

        values = {}

        left, right = st.columns(2)

        for index, feature in enumerate(model.feature_names_in_):

            container = left if index % 2 == 0 else right

            with container:

                values[feature] = st.number_input(
                    label=feature,
                    min_value=1,
                    max_value=100,
                    value=1,
                    step=1
                )

        submitted = st.form_submit_button(
            "Predict Diabetes",
            type="primary",
            width="stretch"
        )

    if submitted:

        input_data = pd.DataFrame(
            [[values[feature] for feature in model.feature_names_in_]],
            columns=model.feature_names_in_
        )

        try:

            prediction = int(model.predict(input_data)[0])

            probability = float(
                model.predict_proba(input_data)[0][1]
            )

            st.markdown("---")
            st.subheader("Prediction Result")

            if prediction == 1:

                st.error(
                    "The model predicts a positive diabetes indicator."
                )

            else:

                st.success(
                    "The model predicts a negative diabetes indicator."
                )

            st.metric(
                "Estimated Positive-Class Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(probability)

            st.warning("""
            This is an educational machine-learning estimate,
            not a medical diagnosis. The requested 1–100 inputs
            may not match the original dataset encoding.
            """)

        except Exception as error:

            st.error("Prediction failed.")
            st.exception(error)


# ==========================================
# DATASET ANALYSIS
# ==========================================

elif page == "Dataset Analysis":

    st.title("📊 Dataset Analysis")

    try:

        df = load_dataset()

        col1, col2, col3 = st.columns(3)

        col1.metric("Test Records", f"{len(df):,}")
        col2.metric("Total Columns", len(df.columns))

        col3.metric(
            "Input Features",
            len(model.feature_names_in_)
        )

        st.markdown("---")
        st.subheader("Diabetes Class Distribution")

        counts = df["Diabetes_binary"].value_counts().sort_index()

        chart_data = pd.DataFrame({
            "Class": [
                "Non-diabetic (0)",
                "Diabetes indicator (1)"
            ],
            "Records": [
                int(counts.get(0, 0)),
                int(counts.get(1, 0))
            ]
        })

        st.bar_chart(chart_data.set_index("Class"))

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(20),
            width="stretch"
        )

        st.subheader("Statistical Summary")

        st.dataframe(
            df.describe().T,
            width="stretch"
        )

    except Exception as error:

        st.error("Dataset could not be loaded.")
        st.exception(error)


# ==========================================
# MODEL COMPARISON
# ==========================================

elif page == "Model Comparison":

    st.title("📈 Model Comparison")

    comparison = load_result("model_comparison.csv")

    if comparison is not None and not comparison.empty:

        st.subheader("Model Performance Results")

        st.dataframe(
            comparison,
            width="stretch"
        )

        model_column = next(
            (
                col for col in comparison.columns
                if str(col).strip().lower() == "model"
            ),
            comparison.columns[0]
        )

        chart_data = comparison.copy()

        chart_data[model_column] = (
            chart_data[model_column].astype(str)
        )

        chart_data = chart_data.set_index(model_column)

        chart_data = chart_data.select_dtypes(
            include="number"
        )

        if not chart_data.empty:

            st.subheader("Performance Chart")
            st.bar_chart(chart_data)

    else:

        st.info("Model comparison CSV is not available.")

    image_path = RESULTS / "model_comparison.png"

    if image_path.exists():

        st.subheader("Comparison Visualization")

        st.image(
            str(image_path),
            width="stretch"
        )

    st.caption(
        "The models used different train-test splits, "
        "so this comparison is preliminary."
    )


# ==========================================
# EXPLAINABLE AI WITH SHAP
# ==========================================

elif page == "Explainable AI":

    st.title("🧠 Explainable AI")

    st.write("""
    Explainable AI helps us understand how the Random Forest model
    uses health indicators when producing predictions.
    """)

    # --------------------------------------
    # RANDOM FOREST FEATURE IMPORTANCE
    # --------------------------------------

    st.subheader("1. Random Forest Feature Importance")

    importance = load_result("feature_importance.csv")

    if importance is not None and not importance.empty:

        st.dataframe(
            importance,
            width="stretch"
        )

        feature_column = importance.columns[0]

        numeric_columns = importance.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:

            value_column = numeric_columns[0]

            plot_data = importance.sort_values(
                value_column,
                ascending=False
            ).head(15)

            st.bar_chart(
                plot_data.set_index(feature_column)[value_column]
            )

    else:

        st.info(
            "Feature importance file is not available. "
            "Run python src/explain_model.py to generate it."
        )

    st.caption(
        "Feature importance describes how the model uses features. "
        "It does not establish cause and effect."
    )

    st.markdown("---")

    # --------------------------------------
    # SHAP GLOBAL EXPLANATION
    # --------------------------------------

    st.subheader("2. SHAP Global Explanation")

    shap_importance = load_result("shap_feature_importance.csv")

    if shap_importance is not None and not shap_importance.empty:

        st.write("""
        Mean absolute SHAP values indicate how strongly each feature
        contributes to the model output on average across the sampled
        test records.
        """)

        st.dataframe(
            shap_importance,
            width="stretch"
        )

        shap_summary_path = RESULTS / "shap_summary.png"

        if shap_summary_path.exists():

            st.image(
                str(shap_summary_path),
                caption="SHAP Summary Plot",
                width="stretch"
            )

        else:

            st.info("SHAP summary image is not available.")

    else:

        st.info(
            "SHAP importance file is not available. "
            "Run python src/explain_model.py to generate SHAP results."
        )

    st.markdown("---")

    # --------------------------------------
    # SHAP INDIVIDUAL EXPLANATION
    # --------------------------------------

    st.subheader("3. Individual Prediction Explanation")

    local_file = RESULTS / "shap_local_explanation.csv"
    local_plot = RESULTS / "shap_local_explanation.png"

    if local_file.exists():

        local_data = pd.read_csv(local_file)

        st.write("""
        This explanation shows the feature contributions for one
        selected test-record prediction.
        """)

        st.dataframe(
            local_data,
            width="stretch"
        )

        if local_plot.exists():

            st.image(
                str(local_plot),
                caption="Individual SHAP Contribution Chart",
                width="stretch"
            )

        else:

            st.info("Individual SHAP chart is not available.")

    else:

        st.info(
            "Individual SHAP explanation is not available. "
            "Run python src/explain_model.py to generate it."
        )

    st.markdown("---")

    st.caption("""
    SHAP explains the behaviour of the machine-learning model.
    It does not establish medical causation or provide a clinical diagnosis.
    """)


# ==========================================
# FAIRNESS ANALYSIS
# ==========================================

elif page == "Fairness Analysis":

    st.title("⚖️ Fairness Analysis")

    st.write("""
    This section evaluates model performance across demographic
    groups and compares the baseline model with the age-adjusted
    fairness experiment.
    """)

    # ======================================
    # ORIGINAL FAIRNESS AUDIT
    # ======================================

    st.header("1. Original Fairness Audit")

    files = [
        ("fairness_by_sex.csv", "Performance by Sex"),
        ("fairness_by_age.csv", "Performance by Age Group")
    ]

    for filename, heading in files:

        st.subheader(heading)

        result = load_result(filename)

        if result is not None and not result.empty:

            st.dataframe(
                result,
                width="stretch"
            )

        else:

            st.info(filename + " is not available.")

    st.markdown("---")

    # ======================================
    # BASELINE VS AGE-ADJUSTED COMPARISON
    # ======================================

    st.header("2. Baseline vs Age-Adjusted Model")

    comparison = load_result(
        "fairness_improvement_comparison.csv"
    )

    if comparison is not None and not comparison.empty:

        # Overall performance

        overall = comparison[
            comparison["Group_Type"] == "Overall"
        ].copy()

        if not overall.empty:

            st.subheader("Overall Performance")

            display_columns = [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1"
            ]

            st.dataframe(
                overall[display_columns].set_index("Model"),
                width="stretch"
            )

            chart_data = overall.set_index("Model")[
                ["Accuracy", "Precision", "Recall", "F1"]
            ]

            st.subheader("Overall Metrics Comparison")

            st.bar_chart(chart_data)

        else:

            st.info("Overall comparison results are unavailable.")

        st.markdown("---")

        # ==================================
        # AGE GROUP RECALL COMPARISON
        # ==================================

        st.subheader("Recall Comparison by Age Group")

        age_results = comparison[
            comparison["Group_Type"] == "Age"
        ].copy()

        if not age_results.empty:

            recall_chart = age_results.pivot(
                index="Group",
                columns="Model",
                values="Recall"
            )

            st.bar_chart(recall_chart)

            st.subheader("Detailed Age Group Results")

            st.dataframe(
                age_results[
                    [
                        "Group",
                        "Model",
                        "Records",
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1"
                    ]
                ],
                width="stretch"
            )

            baseline = age_results[
                age_results["Model"] == "Baseline"
            ]

            adjusted = age_results[
                age_results["Model"] == "Age-Adjusted"
            ]

            if not baseline.empty and not adjusted.empty:

                baseline_gap = (
                    baseline["Recall"].max()
                    - baseline["Recall"].min()
                )

                adjusted_gap = (
                    adjusted["Recall"].max()
                    - adjusted["Recall"].min()
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Baseline Recall Gap",
                    f"{baseline_gap * 100:.2f} pp"
                )

                col2.metric(
                    "Age-Adjusted Recall Gap",
                    f"{adjusted_gap * 100:.2f} pp",
                    delta=f"{(adjusted_gap - baseline_gap) * 100:.2f} pp",
                    delta_color="inverse"
                )

                st.info(
                    "A smaller recall gap indicates more similar "
                    "recall across the evaluated age groups. "
                    "It does not establish overall fairness."
                )

        else:

            st.info("Age-group comparison is unavailable.")

        st.markdown("---")

        # ==================================
        # SEX GROUP COMPARISON
        # ==================================

        st.subheader("Performance Comparison by Sex")

        sex_results = comparison[
            comparison["Group_Type"] == "Sex"
        ].copy()

        if not sex_results.empty:

            st.dataframe(
                sex_results[
                    [
                        "Group",
                        "Model",
                        "Records",
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1"
                    ]
                ],
                width="stretch"
            )

            sex_chart = sex_results.pivot(
                index="Group",
                columns="Model",
                values="Recall"
            )

            st.subheader("Recall by Sex")

            st.bar_chart(sex_chart)

        else:

            st.info("Sex-group comparison is unavailable.")

    else:

        st.info(
            "Fairness improvement results are not available. "
            "Run python src/fairness_improvement.py first."
        )

    st.markdown("---")

    # ======================================
    # AGE-SPECIFIC THRESHOLDS
    # ======================================

    st.header("3. Age-Specific Thresholds")

    thresholds = load_result("age_group_thresholds.csv")

    if thresholds is not None and not thresholds.empty:

        st.write("""
        The following decision thresholds were selected using
        the validation dataset for the fairness experiment.
        """)

        st.dataframe(
            thresholds,
            width="stretch"
        )

    else:

        st.info("Age-specific threshold results are unavailable.")

    st.markdown("---")

    # ======================================
    # RESEARCH FINDINGS
    # ======================================

    st.header("4. Research Findings")

    st.write("""
    The experiment investigated whether age-specific decision
    thresholds could reduce differences in recall across age groups.

    The results showed that:

    - The age-adjusted model improved overall accuracy.
    - Recall differences between age groups decreased substantially.
    - Overall recall and F1-score decreased slightly.
    - Precision and recall trade-offs differed across demographic groups.

    These findings demonstrate a fairness-performance trade-off.
    Further validation is required before considering this approach
    for any real healthcare application.
    """)

    st.warning("""
    Research limitation: The experiment uses age-specific thresholds.
    Since age is a sensitive demographic attribute, this approach
    requires careful ethical, legal and clinical evaluation.
    It is an academic experiment, not a clinical recommendation.
    """)

    st.caption("""
    The baseline and age-adjusted results in this experiment use
    the same experiment model and test set. The experiment model
    uses a separate validation split for threshold selection.
    Therefore, its baseline values may differ from the original
    fairness audit results.
    """)

    st.caption(
        "Fairness metrics describe differences in model performance. "
        "They do not independently establish discrimination or causation."
    )


# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.title("ℹ️ About the Project")

    st.subheader(
        "Disease Prediction Using Hive and Apache Mahout"
    )

    st.write("""
    This academic research project explores diabetes
    indicator prediction using health-related data.

    ### Technologies Used

    - Apache Hive: Big-data analysis and storage.
    - Apache Mahout: Logistic regression.
    - Python and Scikit-learn: Random Forest.
    - Explainable AI: Random Forest feature importance and SHAP.
    - Fairness Analysis: Group-level evaluation.
    - Federated Learning: Three simulated hospitals.

    ### Research Limitations

    The models are not clinically validated.

    Federated learning is simulated and does not implement
    production-grade privacy guarantees.
    """)

    st.success(
        "This project demonstrates the integration of "
        "big-data technologies and machine learning."
    )
