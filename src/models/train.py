# src/models/train.py

from xgboost import XGBClassifier
import joblib
from pathlib import Path


def train_model(X_train, y_train):
    """
    Train XGBoost classifier and save feature columns.
    """

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    # save feature columns
    Path("models").mkdir(exist_ok=True)
    joblib.dump(X_train.columns.tolist(), "models/feature_columns.pkl")

    return model