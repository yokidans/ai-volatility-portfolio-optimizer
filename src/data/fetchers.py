import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from src.config import settings
from src.infra.logging import logger
import pickle

class DataFetcher:
    """Financial data fetcher with preloaded data fallback."""
    
    def __init__(self):
        self.cache_dir = Path(settings.RAW_DATA_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.preloaded_dir = Path(settings.PRELOADED_DATA_DIR)
        self.preloaded_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize preloaded data
        self._initialize_preloaded_data()
    
    def _create_sample_data(self, ticker: str, days: int = 3287) -> pd.DataFrame:
        """Create sample data for a ticker with consistent lengths."""
        dates = pd.date_range(start='2010-01-01', periods=days)
        base_price = np.random.uniform(30, 50, 1)[0]  # Different base price for each ticker
        
        return pd.DataFrame({
            'date': dates,
            'open': base_price + np.cumsum(np.random.normal(0, 0.5, days)),
            'high': base_price + np.cumsum(np.random.normal(0.1, 0.5, days)),
            'low': base_price + np.cumsum(np.random.normal(-0.1, 0.5, days)),
            'close': base_price + np.cumsum(np.random.normal(0, 0.5, days)),
            'volume': np.random.randint(1000000, 100000000, days),
            'ticker': ticker
        })
    
    def _initialize_preloaded_data(self):
        """Create preloaded data files if they don't exist."""
        tickers = settings.TICKERS + settings.BENCHMARKS
        
        for ticker in tickers:
            file_path = self.preloaded_dir / f"{ticker}.pkl"
            if not file_path.exists():
                try:
                    sample_data = self._create_sample_data(ticker)
                    sample_data.to_pickle(file_path)
                    logger.info(f"Created preloaded data for {ticker}")
                except Exception as e:
                    logger.error(f"Failed to create preloaded data for {ticker}: {str(e)}")
    
    def _load_preloaded_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load preloaded data for a ticker."""
        file_path = self.preloaded_dir / f"{ticker}.pkl"
        if file_path.exists():
            try:
                data = pd.read_pickle(file_path)
                # Filter to our date range
                data['date'] = pd.to_datetime(data['date'])
                data = data[(data['date'] >= pd.to_datetime(settings.TRAIN_START)) & 
                           (data['date'] <= pd.to_datetime(settings.END_DATE))]
                if not data.empty:
                    logger.warning(f"Using preloaded data for {ticker}")
                    return data
            except Exception as e:
                logger.error(f"Failed to load preloaded data for {ticker}: {str(e)}")
        return None
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    def _try_yfinance(self, ticker: str) -> Optional[pd.DataFrame]:
        """Try fetching data using yfinance."""
        try:
            data = yf.download(
                ticker,
                start=settings.TRAIN_START - timedelta(days=1),
                end=settings.END_DATE + timedelta(days=1),
                progress=False
            )
            if not data.empty:
                data = data.reset_index()
                data['ticker'] = ticker
                return data.rename(columns={'Date': 'date'})
            return None
        except Exception as e:
            logger.warning(f"yfinance failed for {ticker}: {str(e)}")
            return None
    
    def fetch_stock_data(self, ticker: str) -> pd.DataFrame:
        """Fetch data for a single ticker."""
        # Try cache first
        cache_file = self.cache_dir / f"{ticker}.pkl"
        if cache_file.exists():
            try:
                data = pd.read_pickle(cache_file)
                logger.info(f"Loaded cached data for {ticker}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache for {ticker}: {str(e)}")
        
        logger.info(f"Fetching data for {ticker}")
        
        # Try yfinance
        data = self._try_yfinance(ticker)
        
        # If API fails, use preloaded data
        if data is None:
            data = self._load_preloaded_data(ticker)
        
        if data is None:
            raise ValueError(f"All data sources failed for {ticker}")
        
        # Save to cache
        try:
            data.to_pickle(cache_file)
        except Exception as e:
            logger.warning(f"Failed to save cache for {ticker}: {str(e)}")
        
        return data
    
    def fetch_all_stock_data(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch data for multiple tickers."""
        all_data = []
        for ticker in tickers:
            try:
                data = self.fetch_stock_data(ticker)
                all_data.append(data)
            except Exception as e:
                logger.error(f"Skipping {ticker}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("No stock data could be fetched")
        
        return pd.concat(all_data).sort_values(['date', 'ticker'])

    def fetch_all_data(self) -> pd.DataFrame:
        """Fetch all required data."""
        return self.fetch_all_stock_data(
            tickers=settings.TICKERS + settings.BENCHMARKS
        )

if __name__ == "__main__":
    try:
        fetcher = DataFetcher()
        assets = fetcher.fetch_all_data()
        print("Assets data (5 rows):")
        print(assets.head())
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        raise