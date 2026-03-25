# src/models/predict.py

import joblib
import pandas as pd
import shap
from pathlib import Path


def load_model(model_path: str):
    path = Path(model_path)

    if not path.exists() or path.stat().st_size == 0:
        raise ValueError("Model missing or corrupted")

    return joblib.load(path)


def load_feature_columns(path: str = "models/feature_columns.pkl"):
    p = Path(path)

    if not p.exists() or p.stat().st_size == 0:
        raise ValueError("Feature columns missing")

    return joblib.load(p)


def align_features(input_df: pd.DataFrame, reference_columns):
    input_encoded = pd.get_dummies(input_df)

    for col in reference_columns:
        if col not in input_encoded:
            input_encoded[col] = 0

    return input_encoded[reference_columns]


def generate_reasoning_and_suggestions(shap_values, X_row):
    impacts = pd.DataFrame({
        "feature": X_row.index,
        "value": X_row.values,
        "impact": shap_values
    })

    impacts["abs_impact"] = impacts["impact"].abs()
    impacts = impacts.sort_values(by="abs_impact", ascending=False)

    top = impacts.head(5)

    reasons = []
    suggestions = []

    for _, row in top.iterrows():
        feature = row["feature"]
        impact = row["impact"]

        # ===== Reason =====
        if impact > 0:
            reasons.append(f"{feature} increased churn risk")
        else:
            reasons.append(f"{feature} reduced churn risk")

        # ===== Suggestions (Business Logic) =====
        if "Contract_Month-to-month" in feature and impact > 0:
            suggestions.append("Offer discounts for long-term contracts")

        elif "tenure" in feature and impact < 0:
            suggestions.append("Reward long-term customers with loyalty benefits")

        elif "MonthlyCharges" in feature and impact > 0:
            suggestions.append("Provide more value or reduce pricing plans")

        elif "TechSupport" in feature and impact > 0:
            suggestions.append("Improve customer support quality and availability")

        elif "OnlineSecurity" in feature and impact > 0:
            suggestions.append("Promote security features to increase trust")

        elif "PaymentMethod_Electronic check" in feature and impact > 0:
            suggestions.append("Encourage automatic payment methods")

        elif "InternetService_Fiber optic" in feature and impact > 0:
            suggestions.append("Improve service reliability or pricing")

    # remove duplicates
    suggestions = list(set(suggestions))

    return reasons, suggestions


def predict_with_explanation(model, input_df: pd.DataFrame, reference_columns):
    X = align_features(input_df, reference_columns)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_row = shap_values[0]

    reasons, suggestions = generate_reasoning_and_suggestions(
        shap_row,
        X.iloc[0]
    )

    return {
        "prediction": int(pred),
        "probability": float(prob),
        "reasons": reasons,
        "suggestions": suggestions
    }