from sklearn.ensemble import RandomForestClassifier
from intelligence.training.feature_builder import build_features
from intelligence.training.data_loader import download_nse_data
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os

def train_classifier_model(tickers: list[str], threshold=0.02):
    dfs = []
    for ticker in tickers:
        try:
            df = build_features(download_nse_data([ticker])[ticker])
            df["target"] = (df["Target_5D_Return"] > threshold).astype(int)
            dfs.append(df)
        except Exception as e:
            print(f"Classifier skip {ticker}: {e}")
    full_df = pd.concat(dfs)

    X = full_df[["SMA_10", "SMA_30", "Momentum_10"]]
    y = full_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, "intelligence/models/classifier.pkl")
    print("✅ Classifier model saved.")
