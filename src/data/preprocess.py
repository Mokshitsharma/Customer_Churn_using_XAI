# src/data/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(df: pd.DataFrame):
    """
    Clean and prepare dataset for modeling.
    """

    df = df.copy()

    # drop ID
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # handle missing
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # separate features/target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # one-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test