"""Fetch historical stock data from Yahoo Finance with robust Indian market fallback."""
from __future__ import annotations
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

def fetch_stock_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    # Clean spaces out immediately (e.g. "TATA MOTORS" -> "TATAMOTORS")
    ticker = ticker.strip().upper().replace(" ", "")
    
    is_indian = False
    if not ticker.endswith('.NS') and not ticker.endswith('.BO'):
        us_tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'NFLX', 'AMD']
        if ticker not in us_tickers and len(ticker) >= 3:
            ticker = f"{ticker}.NS"
            is_indian = True
    elif ticker.endswith('.NS') or ticker.endswith('.BO'):
        is_indian = True

    print(f"🔄 Fetching data engine executing for: {ticker}")
    df = pd.DataFrame()

    # 1. Try Yahoo Finance first
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    except Exception:
        pass

    # Clean Yahoo multi-index if it exists
    if not df.empty and isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. PERMANENT FALLBACK: If Yahoo fails/blocks (returns empty) and it's an Indian stock
    if (df.empty or 'Close' not in df.columns) and is_indian:
        print(f"⚠️ Yahoo Finance blocked/failed for {ticker}. Switching to NSE India Direct Fallback...")
        try:
            from jugaad_data.nse import stock_df
            base_ticker = ticker.split('.')[0] # Remove .NS for NSE API
            
            # Calculate dates using date() instead of datetime() to prevent Windows file name errors
            end = date.today()
            days_back = 730 if period == "2y" else 1825 
            start = end - timedelta(days=days_back)
            
            # Fetch straight from NSE
            raw_df = stock_df(symbol=base_ticker, from_date=start, to_date=end, series="EQ")
            
            if not raw_df.empty:
                df = raw_df[['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']].copy()
                df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df.sort_index(ascending=True, inplace=True)
                print(f"🔥 Clean data successfully pulled from NSE India for {base_ticker}!")
        except ImportError:
            print("❌ ERROR: 'jugaad-data' is not installed! Run: pip install jugaad-data")
        except Exception as e:
            print(f"⚠️ NSE Fallback also failed: {e}")

    # Final cleanup before sending to the ML model
    if not df.empty and 'Close' in df.columns:
        df = df.dropna()
        df.index = pd.to_datetime(df.index)
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[[col for col in required_cols if col in df.columns]]
        return df
        
    print(f"❌ Critical Error: All data pipelines failed for {ticker}.")
    return pd.DataFrame()

if __name__ == "__main__":
    data = fetch_stock_data("TATAMOTORS", period="2y")
    print(data.tail())