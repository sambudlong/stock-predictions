import yfinance as yf

data = yf.download("AAPL", start="2025-01-01", end = "2025-07-31", interval="1d")
print(data.head())
