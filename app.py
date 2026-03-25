# app.py

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.models.predict import (
    load_model,
    load_feature_columns,
    predict_with_explanation,
    align_features
)

MODEL_PATH = "models/churn_model.pkl"


@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    feature_columns = load_feature_columns()
    return model, feature_columns


def user_input():
    return {
        "gender": st.selectbox("Gender", ["Male", "Female"]),
        "SeniorCitizen": st.selectbox("Senior Citizen", [0, 1]),
        "Partner": st.selectbox("Partner", ["Yes", "No"]),
        "Dependents": st.selectbox("Dependents", ["Yes", "No"]),
        "tenure": st.slider("Tenure (months)", 0, 72, 1),
        "PhoneService": st.selectbox("Phone Service", ["Yes", "No"]),
        "MultipleLines": st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"]),
        "InternetService": st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": st.selectbox("Online Security", ["Yes", "No"]),
        "OnlineBackup": st.selectbox("Online Backup", ["Yes", "No"]),
        "DeviceProtection": st.selectbox("Device Protection", ["Yes", "No"]),
        "TechSupport": st.selectbox("Tech Support", ["Yes", "No"]),
        "StreamingTV": st.selectbox("Streaming TV", ["Yes", "No"]),
        "StreamingMovies": st.selectbox("Streaming Movies", ["Yes", "No"]),
        "Contract": st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": st.selectbox("Paperless Billing", ["Yes", "No"]),
        "PaymentMethod": st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        ),
        "MonthlyCharges": st.number_input("Monthly Charges", 0.0, 200.0, 50.0),
        "TotalCharges": st.number_input("Total Charges", 0.0, 10000.0, 100.0),
    }


def main():
    st.set_page_config(page_title="Customer Churn XAI", layout="wide")

    st.title("Customer Churn Prediction with Explainable AI")

    model, feature_columns = load_artifacts()

    input_data = user_input()

    if st.button("Predict Churn"):
        input_df = pd.DataFrame([input_data])

        result = predict_with_explanation(
            model,
            input_df,
            feature_columns
        )

        # ===== Prediction =====
        st.subheader("Prediction")
        st.write("Churn:", "Yes" if result["prediction"] == 1 else "No")
        st.write("Probability:", round(result["probability"], 4))

        # ===== Reasons =====
        st.subheader("Why this prediction? (Explainable AI)")
        for reason in result["reasons"]:
            st.write("•", reason)

        # ===== Suggestions =====
        st.subheader("How to Reduce Churn (Actionable Insights)")
        if result["suggestions"]:
            for suggestion in result["suggestions"]:
                st.write("✅", suggestion)
        else:
            st.write("No major risk factors detected. Customer is stable.")

        # ===== SHAP Visuals =====
        st.subheader("SHAP Explanation (Visual)")

        X = align_features(input_df, feature_columns)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # ---- Waterfall Plot ----
        fig_waterfall, ax = plt.subplots()
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value,
            shap_values[0],
            X.iloc[0]
        )
        st.pyplot(fig_waterfall)
        plt.close(fig_waterfall)

        # ---- Feature Importance Bar ----
        fig_bar, ax = plt.subplots()
        shap.summary_plot(
            shap_values,
            X,
            plot_type="bar",
            show=False
        )
        st.pyplot(fig_bar)
        plt.close(fig_bar)


if __name__ == "__main__":
    main()