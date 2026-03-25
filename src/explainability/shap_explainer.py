# src/explainability/shap_explainer.py

import shap
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def generate_shap_explanations(model, X_test, output_dir: str):
    """
    Generate and save SHAP explanations.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # initialize explainer
    explainer = shap.TreeExplainer(model)

    # compute SHAP values
    shap_values = explainer.shap_values(X_test)

    # ===== Summary Plot =====
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(output_path / "summary_plot.png", bbox_inches="tight")
    plt.close()

    # ===== Bar Plot (feature importance) =====
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(output_path / "feature_importance.png", bbox_inches="tight")
    plt.close()

    # ===== Force Plot (single prediction) =====
    sample_index = 0
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[sample_index],
        X_test.iloc[sample_index],
        matplotlib=True
    )

    plt.savefig(output_path / "force_plot.png", bbox_inches="tight")
    plt.close()

    # ===== Save raw SHAP values =====
    np.save(output_path / "shap_values.npy", shap_values)