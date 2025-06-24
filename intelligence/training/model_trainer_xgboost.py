# intelligence/training/model_trainer_xgboost.py

from intelligence.training.data_loader import download_nse_data
from intelligence.training.feature_builder import build_features
import xgboost as xgb
import joblib
import os
import pandas as pd


from sklearn.model_selection import train_test_split

def train_model(tickers: list[str]):
    all_data = []
    
    for ticker in tickers:
        try:
            prices = download_nse_data([ticker])[ticker]
            df = build_features(prices)
            df["ticker"] = ticker
            all_data.append(df)
        except Exception as e:
            print(f"Skipping {ticker}: {e}")
    
    full_df = pd.concat(all_data)
    X = full_df[["SMA_10", "SMA_30", "Momentum_10"]]
    y = full_df["Target_5D_Return"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    os.makedirs("intelligence/models", exist_ok=True)
    joblib.dump(model, "intelligence/models/xgboost_model.pkl")
    print("✅ XGBoost model saved to intelligence/models/xgboost_model.pkl")
