import yfinance as yf
import pandas as pd
from pathlib import Path

# Tickers
tickers = ["AAPL", "MSFT", "GOOGL"]

# Download adjusted close prices (force Adj Close to exist)
data = yf.download(tickers, start="2018-01-01", end="2023-01-01", auto_adjust=False)["Adj Close"]

# Compute daily returns
returns = data.pct_change().dropna()

# Ensure folder exists
Path("data").mkdir(exist_ok=True)

# Save to CSV
returns.to_csv("data/returns.csv")

print("✅ Data downloaded and saved to data/returns.csv")
print(returns.head())
