"""Forecast future stock prices using trained or freshly-trained models."""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import joblib

from data_loader import fetch_stock_data
from features import add_technical_indicators, FEATURE_COLS
from models import get_model, evaluate

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _train_quick(df, name):
    df_feat = add_technical_indicators(df)
    df_feat["target"] = df_feat["Close"].shift(-1)
    df_feat = df_feat.dropna()
    X = df_feat[FEATURE_COLS].values
    y = df_feat["target"].values
    split = int(len(X) * 0.8)
    model = get_model(name)
    model.fit(X[:split], y[:split])
    metrics = evaluate(y[split:], model.predict(X[split:]))
    return model, metrics


def forecast(ticker: str, days: int = 7, model_name: str = "random_forest"):
    df = fetch_stock_data(ticker, period="2y")
    path = os.path.join(MODELS_DIR, f"{ticker.upper()}_{model_name}.pkl")

    if os.path.exists(path):
        model = joblib.load(path)
        # Need metrics — compute on holdout
        _, metrics = _train_quick(df, model_name)
    else:
        model, metrics = _train_quick(df, model_name)

    history = df.copy()
    predictions = []
    dates = []

    for _ in range(days):
        feat_df = add_technical_indicators(history)
        if feat_df.empty:
            break
        x_last = feat_df[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
        next_price = float(model.predict(x_last)[0])

        next_date = history.index[-1] + pd.Timedelta(days=1)
        # Skip weekends
        while next_date.weekday() >= 5:
            next_date += pd.Timedelta(days=1)

        new_row = pd.DataFrame({
            "Open": [next_price], "High": [next_price],
            "Low": [next_price], "Close": [next_price],
            "Volume": [history["Volume"].iloc[-20:].mean()],
        }, index=[next_date])
        history = pd.concat([history, new_row])
        predictions.append(next_price)
        dates.append(next_date)

    return {
        "ticker": ticker.upper(),
        "model": model_name,
        "metrics": metrics.to_dict(),
        "history": df.tail(90).reset_index().rename(columns={"index": "Date"}),
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in dates],
        "forecast_prices": [round(p, 2) for p in predictions],
        "last_close": round(float(df["Close"].iloc[-1]), 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--model", default="random_forest")
    args = parser.parse_args()
    result = forecast(args.ticker, args.days, args.model)
    print(f"\n📊 {result['ticker']} — Model: {result['model']}")
    print(f"   Metrics: {result['metrics']}")
    print(f"   Last close: ${result['last_close']}")
    print(f"\n🔮 Forecast next {args.days} days:")
    for d, p in zip(result["forecast_dates"], result["forecast_prices"]):
        print(f"   {d}  →  ${p}")
