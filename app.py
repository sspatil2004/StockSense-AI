"""Flask web dashboard for Stock Price Prediction with Indian Market Optimization."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import yfinance as yf
from flask import Flask, render_template, request, jsonify
from predict import forecast

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    
    # Strip spaces to prevent "tata motors" spacing errors
    ticker = data.get("ticker", "AAPL").strip().upper().replace(" ", "")
    days = int(data.get("days", 7))
    model = data.get("model", "random_forest")
    
    # Standardize ticker names to support seamless Indian company lookup
    ticker_clean = ticker
    is_indian = False
    if not ticker_clean.endswith('.NS') and not ticker_clean.endswith('.BO'):
        us_tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'NFLX', 'AMD']
        if ticker_clean not in us_tickers and len(ticker_clean) >= 3:
            ticker_clean = f"{ticker_clean}.NS"
            is_indian = True
    elif ticker_clean.endswith('.NS') or ticker_clean.endswith('.BO'):
        is_indian = True

    try:
        # Run core ML forecasting loops
        result = forecast(ticker, days=days, model_name=model)
        hist = result["history"]
        date_col = "Date" if "Date" in hist.columns else hist.columns[0]
        
        # Initialize ticker object using clean formatted string for financial extraction
        ticker_obj = yf.Ticker(ticker_clean)
        
        # --- 100% RELIABLE HIGH/LOW FETCH (From our own dataframe to avoid Yahoo blocks) ---
        try:
            day_high = round(float(hist["High"].iloc[-1]), 2)
            day_low = round(float(hist["Low"].iloc[-1]), 2)
        except Exception:
            day_high, day_low = 'N/A', 'N/A'

        # --- FETCH MARKET CAP SEPARATELY (Using lightweight fast_info for cloud servers) ---
        try:
            # fast_info bypasses the heavy rate limits that block .info on Render
            raw_market_cap = ticker_obj.fast_info['marketCap']
            market_cap_cr = round(raw_market_cap / 10000000, 2) if raw_market_cap else 'N/A'
        except Exception:
            try:
                # Backup attempt just in case
                ticker_info = ticker_obj.info
                raw_market_cap = ticker_info.get('marketCap', 0)
                market_cap_cr = round(raw_market_cap / 10000000, 2) if raw_market_cap else 'N/A'
            except Exception:
                market_cap_cr = 'N/A'

        # --- FETCH QUARTERLY REVENUE STATEMENTS ---
        revenue_labels = []
        revenue_values = []
        latest_revenue_str = "N/A"
        
        try:
            income_stmt = ticker_obj.quarterly_income_stmt
            if income_stmt is not None and "Total Revenue" in income_stmt.index:
                rev_row = income_stmt.loc["Total Revenue"]
                
                # Take the last 4 reported quarters and reverse them to be chronological (oldest to newest)
                revenue_labels = [str(c)[:10] for c in rev_row.index[:4]][::-1]
                revenue_values = [round(float(v) / 10000000, 2) for v in rev_row.values[:4]][::-1]
                
                # Format a neat summary string of the latest revenue amount for display over the graph
                if len(revenue_values) > 0:
                    latest_val = revenue_values[-1]
                    latest_revenue_str = f"₹ {latest_val:,.2f} Cr" if is_indian else f"$ {latest_val:,.2f} M"
        except Exception:
            revenue_labels, revenue_values = [], []
            latest_revenue_str = "N/A"

        return jsonify({
            "ok": True,
            "ticker": result["ticker"],
            "model": result["model"],
            "metrics": result["metrics"],
            "last_close": result["last_close"],
            
            # Metadata payloads
            "day_high": day_high,
            "day_low": day_low,
            "market_cap_cr": market_cap_cr,
            
            # Revenue arrays & latest summary string
            "revenue_labels": revenue_labels,
            "revenue_values": revenue_values,
            "latest_revenue": latest_revenue_str,
            
            "history_dates": [str(d)[:10] for d in hist[date_col].tolist()],
            "history_prices": [round(float(p), 2) for p in hist["Close"].tolist()],
            "forecast_dates": result["forecast_dates"],
            "forecast_prices": result["forecast_prices"],
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)