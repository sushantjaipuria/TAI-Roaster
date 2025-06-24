import joblib
import numpy as np
import os
from pathlib import Path

# Use absolute path resolution to handle different working directories
BASE_DIR = Path(__file__).parent.parent.parent  # Go up to TAI-Roaster root
MODEL_PATH = BASE_DIR / "intelligence" / "training" / "intelligence" / "models" / "enhanced" / "ngboost_model.pkl"

def load_ngboost_model():
    """Load NGBoost model with error handling"""
    try:
        if MODEL_PATH.exists():
            loaded = joblib.load(str(MODEL_PATH))
            print(f"[DEBUG] Loaded NGBoost model from {MODEL_PATH}")
            return loaded
        else:
            print(f"[WARNING] NGBoost model not found at {MODEL_PATH}")
            return None
    except Exception as e:
        print(f"[WARNING] Failed to load NGBoost model: {e}")
        print("[INFO] Using fallback NGBoost prediction")
        return None

# Try to load the model, but don't fail if it doesn't work
model = load_ngboost_model()

def predict_distribution(features):
    """
    Predict return distribution using NGBoost
    Returns dict with 'mean' and 'std'
    """
    try:
        if model is not None:
            # Use the actual model if available
            prediction = model.predict(np.array(features).reshape(1, -1))
            return {
                'mean': float(prediction[0]),
                'std': 0.05  # Default std
            }
        else:
            # Fallback prediction based on features
            feature_sum = sum(features) if features else 0
            base_return = 0.08  # 8% base return
            
            # Simple heuristic based on feature values
            if feature_sum > 0:
                predicted_return = min(0.25, max(-0.15, base_return + feature_sum * 0.001))
            else:
                predicted_return = base_return
            
            return {
                'mean': predicted_return,
                'std': 0.05  # 5% standard deviation
            }
    except Exception as e:
        print(f"[ERROR] NGBoost prediction failed: {e}")
        # Ultimate fallback
        return {
            'mean': 0.08,
            'std': 0.05
        }
