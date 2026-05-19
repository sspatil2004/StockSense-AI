"""Flask web dashboard for Stock Price Prediction."""
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
    ticker = data.get("ticker", "AAPL").strip().upper()
    days = int(data.get("days", 7))
    model = data.get("model", "random_forest")
    try:
        # Run core ML forecasting loops
        result = forecast(ticker, days=days, model_name=model)
        hist = result["history"]
        date_col = "Date" if "Date" in hist.columns else hist.columns[0]
        
        # Initialize ticker object for financial metadata extraction
        ticker_obj = yf.Ticker(ticker)
        
        # --- FETCH HIGH, LOW, AND MARKET CAP ---
        try:
            ticker_info = ticker_obj.info
            day_high = ticker_info.get('dayHigh', 'N/A')
            day_low = ticker_info.get('dayLow', 'N/A')
            raw_market_cap = ticker_info.get('marketCap', 0)
            market_cap_cr = round(raw_market_cap / 10000000, 2) if raw_market_cap else 'N/A'
        except Exception:
            day_high, day_low, market_cap_cr = 'N/A', 'N/A', 'N/A'

        # --- FETCH QUARTERLY REVENUE STATEMENTS ---
        revenue_labels = []
        revenue_values = []
        try:
            income_stmt = ticker_obj.quarterly_income_stmt
            if income_stmt is not None and "Total Revenue" in income_stmt.index:
                rev_row = income_stmt.loc["Total Revenue"]
                # Take the last 4 reported quarters and reverse them to be chronological (oldest to newest)
                revenue_labels = [str(c)[:10] for c in rev_row.index[:4]][::-1]
                revenue_values = [round(float(v) / 10000000, 2) for v in rev_row.values[:4]][::-1]
        except Exception:
            revenue_labels, revenue_values = [], []

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
            
            # Revenue arrays
            "revenue_labels": revenue_labels,
            "revenue_values": revenue_values,
            
            "history_dates": [str(d)[:10] for d in hist[date_col].tolist()],
            "history_prices": [round(float(p), 2) for p in hist["Close"].tolist()],
            "forecast_dates": result["forecast_dates"],
            "forecast_prices": result["forecast_prices"],
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)