# src/models/save_model.py

import joblib
from pathlib import Path


def save_model(model, output_path: str):
    """
    Save trained model.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)