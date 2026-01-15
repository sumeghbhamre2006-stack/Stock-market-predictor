import yfinance as yf

def fetch_stock(symbol: str):
    df = yf.download(symbol, period="5y", interval="1d")
    df.dropna(inplace=True)
    return df
