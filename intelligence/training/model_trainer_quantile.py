from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from intelligence.training.feature_builder import build_features
from intelligence.training.data_loader import download_nse_data
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os

def train_quantile_model(tickers: list[str]):
    print("🚀 Starting quantile model training...")
    dfs = []
    for ticker in tickers:
        try:
            df = build_features(download_nse_data([ticker])[ticker])
            # Simulated quantile targets
            df["Target_5D_p10"] = df["Target_5D_Return"] * 0.8
            df["Target_5D_p50"] = df["Target_5D_Return"]
            df["Target_5D_p90"] = df["Target_5D_Return"] * 1.2
            dfs.append(df)
        except Exception as e:
            print(f"Quantile skip {ticker}: {e}")
    
    if not dfs:
        print("❌ No dataframes loaded. Exiting.")
        return

    full_df = pd.concat(dfs)
    print(f"✅ Loaded {len(full_df)} rows of feature data.")

    X = full_df[["SMA_10", "SMA_30", "Momentum_10"]]
    y = full_df[["Target_5D_p10", "Target_5D_p50", "Target_5D_p90"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    base_model = LGBMRegressor(objective='quantile')
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    joblib.dump(model, "intelligence/models/quantile_model.pkl")
    print("✅ Quantile model saved as MultiOutputRegressor.")

# 👇 This ensures the script runs when called directly or via `-m`
if __name__ == "__main__":
    train_quantile_model(["TCS", "INFY", "HDFCBANK"])
