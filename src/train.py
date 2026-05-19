"""Train and save stock-prediction models."""
from __future__ import annotations
import argparse
import os
import joblib
import numpy as np

from data_loader import fetch_stock_data
from features import add_technical_indicators, FEATURE_COLS
from models import get_model, evaluate

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def prepare_xy(df):
    df = add_technical_indicators(df)
    df["target"] = df["Close"].shift(-1)
    df = df.dropna()
    X = df[FEATURE_COLS].values
    y = df["target"].values
    return X, y, df


def train(ticker: str, period: str = "5y") -> None:
    print(f"📥 Fetching {ticker} ({period})...")
    df = fetch_stock_data(ticker, period=period)
    X, y, _ = prepare_xy(df)

    # Chronological split (no shuffle for time series)
    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    for name in ("linear", "random_forest"):
        print(f"\n🧠 Training {name}...")
        model = get_model(name)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        print(f"   {name} → {metrics.to_dict()}")

        path = os.path.join(MODELS_DIR, f"{ticker.upper()}_{name}.pkl")
        joblib.dump(model, path)
        print(f"   💾 Saved → {path}")

    print("\n✅ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default="5y")
    args = parser.parse_args()
    train(args.ticker, args.period)
