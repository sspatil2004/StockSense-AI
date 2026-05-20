"""Forecast future stock prices using trained or freshly-trained models with variance injection."""
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
    # Clear out formatting strings before looking up tickers locally
    ticker_clean = ticker.strip().upper()
    if not ticker_clean.endswith('.NS') and not ticker_clean.endswith('.BO'):
        us_tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'NFLX', 'AMD']
        if ticker_clean not in us_tickers and len(ticker_clean) >= 3:
            ticker_clean = f"{ticker_clean}.NS"

    df = fetch_stock_data(ticker_clean, period="2y")
    
    if df.empty:
        raise ValueError(f"Could not load data for asset payload: {ticker_clean}")

    path = os.path.join(MODELS_DIR, f"{ticker_clean}_{model_name}.pkl")

    if os.path.exists(path):
        model = joblib.load(path)
        _, metrics = _train_quick(df, model_name)
    else:
        model, metrics = _train_quick(df, model_name)

    history = df.copy()
    predictions = []
    dates = []

    # Calculate 20-day historical standard deviations to generate realistic intraday price spreads
    pct_changes = history["Close"].pct_change().dropna()
    historical_volatility = float(pct_changes.std()) if len(pct_changes) > 0 else 0.015
    
    avg_high_spread = float((history["High"] - history["Close"]).mean())
    avg_low_spread = float((history["Close"] - history["Low"]).mean())
    avg_open_spread = float((history["Open"] - history["Close"].shift(1)).abs().mean())

    for i in range(days):
        feat_df = add_technical_indicators(history)
        if feat_df.empty:
            break
            
        x_last = feat_df[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
        
        # Base model estimation
        predicted_base = float(model.predict(x_last)[0])
        
        # Ingress slight autoregressive noise injection to force variance on multi-day horizons
        if i > 0:
            noise_factor = np.random.normal(0, historical_volatility * 0.3)
            next_price = predicted_base * (1 + noise_factor)
        else:
            next_price = predicted_base

        next_date = history.index[-1] + pd.Timedelta(days=1)
        # Skip weekends cleanly
        while next_date.weekday() >= 5:
            next_date += pd.Timedelta(days=1)

        # Synthesize realistic Open, High, and Low values so moving technical indicators react organically
        simulated_open = next_price + np.random.uniform(-avg_open_spread, avg_open_spread)
        simulated_high = max(next_price, simulated_open) + np.random.uniform(0, avg_high_spread)
        simulated_low = min(next_price, simulated_open) - np.random.uniform(0, avg_low_spread)
        simulated_volume = float(history["Volume"].iloc[-20:].mean() * np.random.uniform(0.8, 1.2))

        new_row = pd.DataFrame({
            "Open": [round(simulated_open, 2)], 
            "High": [round(simulated_high, 2)],
            "Low": [round(simulated_low, 2)], 
            "Close": [round(next_price, 2)],
            "Volume": [round(simulated_volume, 0)],
        }, index=[next_date])
        
        history = pd.concat([history, new_row])
        predictions.append(next_price)
        dates.append(next_date)

    return {
        "ticker": ticker_clean,
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