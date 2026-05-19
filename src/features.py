"""Technical indicators & feature engineering."""
from __future__ import annotations
import numpy as np
import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA, EMA, RSI, MACD, Bollinger Bands, and lag features."""
    out = df.copy()
    close = out["Close"]

    # Moving averages
    out["SMA_10"] = close.rolling(10).mean()
    out["SMA_20"] = close.rolling(20).mean()
    out["SMA_50"] = close.rolling(50).mean()
    out["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    out["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # MACD
    out["MACD"] = out["EMA_12"] - out["EMA_26"]
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    out["BB_upper"] = mid + 2 * std
    out["BB_lower"] = mid - 2 * std

    # Lag features
    for lag in (1, 2, 3, 5, 10):
        out[f"lag_{lag}"] = close.shift(lag)

    # Returns & volatility
    out["return_1d"] = close.pct_change()
    out["volatility_10"] = out["return_1d"].rolling(10).std()

    out = out.dropna()
    return out


FEATURE_COLS = [
    "Open", "High", "Low", "Volume",
    "SMA_10", "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    "MACD", "MACD_signal", "RSI", "BB_upper", "BB_lower",
    "lag_1", "lag_2", "lag_3", "lag_5", "lag_10",
    "return_1d", "volatility_10",
]
