# main.py

import pandas as pd

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data

from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.models.save_model import save_model

from src.explainability.shap_explainer import generate_shap_explanations

from src.models.predict import (
    load_model,
    load_feature_columns,
    predict_with_explanation
)

FILE_PATH = r"C:\Users\dell\OneDrive\Desktop\xai-financial-sentiment\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn_model.pkl"
SHAP_OUTPUT = "reports/shap_outputs"


def train_pipeline():
    df = load_data(FILE_PATH)

    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model, MODEL_PATH)

    generate_shap_explanations(model, X_test, SHAP_OUTPUT)


def predict_pipeline():
    model = load_model(MODEL_PATH)
    feature_columns = load_feature_columns()

    sample = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }

    input_df = pd.DataFrame([sample])

    result = predict_with_explanation(
        model,
        input_df,
        feature_columns
    )

    print("\nPrediction Result:")
    print(result)


if __name__ == "__main__":
    train_pipeline()
    predict_pipeline()