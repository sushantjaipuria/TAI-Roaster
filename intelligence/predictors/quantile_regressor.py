import joblib
import os
import numpy as np
from pathlib import Path

# Use absolute path resolution to handle different working directories
BASE_DIR = Path(__file__).parent.parent.parent  # Go up to TAI-Roaster root
MODEL_PATH = BASE_DIR / "intelligence" / "models" / "enhanced" / "lasso_model.pkl"

def load_quantile_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Quantile model not found at {MODEL_PATH}")

    loaded = joblib.load(str(MODEL_PATH))

    if isinstance(loaded, dict):
        # You previously used a dict — now warn instead of crashing
        print("⚠️ Model loaded as dict, falling back to raw dict object.")
        return loaded.get("model", loaded)

    print(f"[DEBUG] Loaded Quantile model of type {type(loaded)}")
    return loaded

model = load_quantile_model()

def predict_quantiles(features):
    pred = model.predict([features])
    return {
        "p10": pred[0][0],
        "p50": pred[0][1],
        "p90": pred[0][2]
    }
