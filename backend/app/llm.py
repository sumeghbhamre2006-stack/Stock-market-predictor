def analyze_stock(symbol: str, indicators: dict):
    """
    LLM placeholder function.
    Later this will call OpenAI / Gemini / LLaMA.
    """
    rsi = indicators.get("rsi")

    if rsi is None:
        return "Insufficient data for analysis."

    if rsi < 30:
        return f"{symbol} appears oversold (RSI {rsi:.2f}). Possible rebound."
    elif rsi > 70:
        return f"{symbol} appears overbought (RSI {rsi:.2f}). Risk of pullback."
    else:
        return f"{symbol} is trading in a neutral range (RSI {rsi:.2f})."
