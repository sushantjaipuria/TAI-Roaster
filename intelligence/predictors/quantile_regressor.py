import joblib
import os
import numpy as np
from pathlib import Path

# Use absolute path resolution to handle different working directories
BASE_DIR = Path(__file__).parent.parent.parent  # Go up to TAI-Roaster root
MODEL_PATH = BASE_DIR / "intelligence" / "training" / "intelligence" / "models" / "enhanced" / "lasso_model.pkl"

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
    # Lasso model returns a single prediction, simulate quantiles around it
    prediction = pred[0] if isinstance(pred, (list, np.ndarray)) and len(pred) > 0 else float(pred)
    # Create reasonable quantiles around the main prediction
    std_estimate = abs(prediction) * 0.1  # 10% of prediction as standard deviation estimate
    return {
        "p10": prediction - 1.28 * std_estimate,  # ~10th percentile
        "p50": prediction,                        # 50th percentile (median)
        "p90": prediction + 1.28 * std_estimate   # ~90th percentile
    }
