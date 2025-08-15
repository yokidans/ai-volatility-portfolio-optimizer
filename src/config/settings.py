from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Tickers
EQUITY_TICKERS = ["SPY", "TSLA"]
BOND_TICKERS = ["BND"]
VOLATILITY_TICKERS = ["VIX"]
MACRO_TICKERS = ["DGS10", "DGS2"]

# Seeds
RANDOM_SEED = 42
