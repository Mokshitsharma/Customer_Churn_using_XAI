# src/models/evaluate.py

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def evaluate_model(model, X_test, y_test):
    """
    Evaluate classification performance.
    """

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))