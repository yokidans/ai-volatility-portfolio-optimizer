from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
REPO_ROOT = Path(__file__).parent.parent.parent

# Data directories
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
PRELOADED_DATA_DIR = REPO_ROOT / "data" / "preloaded"  # New directory for preloaded data

# Logs directory
LOGS_DIR = REPO_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Data parameters
TICKERS = ["AAPL", "MSFT", "GOOG"]  # Using standard ticker symbols
BENCHMARKS = ["SPY"]  # S&P 500 ETF

# Date ranges
TRAIN_START = datetime(2010, 1, 1)
END_DATE = datetime.now()

# Cache settings
CACHE_FORMAT = "csv"  # Options: "csv", "parquet", "pickle"