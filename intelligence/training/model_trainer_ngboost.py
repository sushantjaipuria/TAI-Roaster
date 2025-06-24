from ngboost import NGBRegressor
from sklearn.model_selection import train_test_split
from intelligence.training.data_loader import download_nse_data
from intelligence.training.feature_builder import build_features
import pandas as pd
import joblib
import os

def train_ngboost_model(tickers: list[str]):
    dfs = []
    for ticker in tickers:
        try:
            df = build_features(download_nse_data([ticker])[ticker])
            dfs.append(df)
        except Exception as e:
            print(f"NGBoost skip {ticker}: {e}")
    full_df = pd.concat(dfs)

    X = full_df[["SMA_10", "SMA_30", "Momentum_10"]]
    y = full_df["Target_5D_Return"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = NGBRegressor()
    model.fit(X_train, y_train)

    os.makedirs("intelligence/models", exist_ok=True)
    joblib.dump(model, "intelligence/models/ngboost_model.pkl")
    print("✅ NGBoost model saved.")
