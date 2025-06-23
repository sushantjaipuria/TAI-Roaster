import joblib
import numpy as np
import os
from pathlib import Path

# Use absolute path resolution to handle different working directories
BASE_DIR = Path(__file__).parent.parent.parent  # Go up to TAI-Roaster root
MODEL_PATH = BASE_DIR / "intelligence" / "models" / "enhanced" / "random_forest_model.pkl"

def load_classifier_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Classifier model not found at {MODEL_PATH}")

    loaded = joblib.load(str(MODEL_PATH))

    # Only accept direct model; do not try to extract 'model' key
    if isinstance(loaded, dict):
        raise ValueError("❌ classifier.pkl is expected to be a model, not a dictionary.")

    print(f"[DEBUG] Loaded Classifier model of type {type(loaded)}")
    return loaded

model = load_classifier_model()

def predict_probability_gt_threshold(features, threshold=0.02):
    proba = model.predict_proba([features])[0][1]
    return proba
