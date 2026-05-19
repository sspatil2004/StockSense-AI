"""Fetch historical stock data from Yahoo Finance."""
from __future__ import annotations
import pandas as pd
import yfinance as yf


def fetch_stock_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV data for a given ticker.

    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'TSLA', 'RELIANCE.NS').
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max).
        interval: Data interval (1d, 1wk, 1mo).

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume.
    """
    # Using Ticker().history is significantly more stable for international markets
    # because it uses an updated API path with native browser tracking fallbacks.
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval, auto_adjust=True)
    except Exception:
        df = pd.DataFrame()

    # Dynamic fallback to standard download if the history engine drops out
    if df.empty:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df


if __name__ == "__main__":
    data = fetch_stock_data("AAPL", period="1y")
    print(data.tail())