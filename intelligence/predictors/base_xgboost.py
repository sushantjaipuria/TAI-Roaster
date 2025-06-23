import xgboost as xgb
import numpy as np
import os
import joblib
from pathlib import Path

# Use absolute path resolution to handle different working directories
BASE_DIR = Path(__file__).parent.parent.parent  # Go up to TAI-Roaster root
MODEL_PATH = BASE_DIR / "intelligence" / "models" / "enhanced" / "xgboost_model.pkl"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    loaded = joblib.load(str(MODEL_PATH))

    if isinstance(loaded, dict):
        if "model" in loaded:
            model = loaded["model"]
        else:
            raise ValueError("❌ 'model' key not found in model bundle dictionary.")
    else:
        model = loaded

    print(f"[DEBUG] Loaded model of type {type(model)} from XGBoost")
    return model

# Load once at module level for speed
model = load_model()

def predict_return_xgboost(features):
    return model.predict([features])[0]
