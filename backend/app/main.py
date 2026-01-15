from fastapi import FastAPI, Query

# Internal imports
from app.data import fetch_stock
from app.indicators import calculate_rsi
from app.llm import analyze_stock

# Create FastAPI app
app = FastAPI(title="Stock LLM Predictor")

# -------------------------------
# Root Endpoint
# -------------------------------
@app.get("/")
def root():
    return {
        "message": "Stock LLM Predictor Backend is running 🚀",
        "status": "OK"
    }

# -------------------------------
# Predict Endpoint
# -------------------------------
@app.get("/predict")
@app.get("/predict")
def predict_stock(
    symbol: str = Query(
        ...,
        examples={
            "default": {
                "summary": "Apple stock",
                "value": "AAPL"
            }
        }
    )
):
    df = fetch_stock(symbol)

    if df is None or df.empty:
        return {"error": "Unable to fetch stock data"}

    df["rsi"] = calculate_rsi(df["Close"])
    latest_rsi = df["rsi"].iloc[-1]

    if latest_rsi != latest_rsi:
        return {"error": "Not enough data to calculate RSI"}

    analysis = analyze_stock(symbol, {"rsi": float(latest_rsi)})

    return {
        "symbol": symbol,
        "rsi": round(float(latest_rsi), 2),
        "analysis": analysis
    }


# -------------------------------
# Optional: Ignore favicon error
# -------------------------------
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return ""
