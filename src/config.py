from datetime import datetime, timedelta

class Settings:
    QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
    TICKERS = ["SPY", "AAPL", "MSFT"]
    BENCHMARKS = ["SPY"]
    ROLLING_WINDOW = 252  # 1 year of trading days

settings = Settings()
