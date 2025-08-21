import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from src.infra.logging import logger


class DataLoader:
    """Enhanced Yahoo Finance data loader with caching and persistence."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache = {}
        self.default_start = (datetime.now() - timedelta(days=5 * 365)).strftime(
            "%Y-%m-%d"
        )
        self.default_end = datetime.now().strftime("%Y-%m-%d")
        self.cache_dir = Path(cache_dir)
        self._init_cache()

    def _init_cache(self) -> None:
        """Initialize cache directory and load existing cache."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Load existing cache files
            for file in self.cache_dir.glob("*.pkl"):
                with open(file, "rb") as f:
                    cache_key = file.stem
                    self.cache[cache_key] = pickle.load(f)

        except Exception as e:
            logger.warning(f"Could not initialize cache: {str(e)}")

    def _save_to_cache(self, cache_key: str, data: pd.Series) -> None:
        """Save data to persistent cache."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Could not save to cache: {str(e)}")

    def load_returns(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.Series:
        """
        Load daily returns for a ticker with enhanced options.

        Parameters
        ----------
            ticker (str): Stock ticker symbol
            start (str): Start date in YYYY-MM-DD format
            end (str): End date in YYYY-MM-DD format
            interval (str): Data interval ('1d', '1wk', '1mo')

        Returns
        -------
            pd.Series: Time series of percentage returns

        """
        try:
            cache_key = f"{ticker}_{start}_{end}_{interval}"

            # Check memory cache first
            if cache_key in self.cache:
                return self.cache[cache_key]

            # Check disk cache
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                    self.cache[cache_key] = data
                    return data

            # Download data from Yahoo Finance
            data = yf.download(
                ticker,
                start=start or self.default_start,
                end=end or self.default_end,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

            if data.empty:
                raise ValueError(f"No data found for {ticker}")

            # Calculate returns
            if interval in ["1d", "1wk"]:
                returns = data["Close"].pct_change().dropna()
            else:
                # For monthly data, use log returns
                returns = np.log(data["Close"]).diff().dropna()

            returns.name = ticker

            # Update caches
            self.cache[cache_key] = returns
            self._save_to_cache(cache_key, returns)

            return returns

        except Exception as e:
            logger.error(f"Failed to load data for {ticker}: {str(e)}", exc_info=True)
            raise

    def get_multiple_returns(self, tickers: list, **kwargs) -> pd.DataFrame:
        """
        Load returns for multiple tickers.

        Parameters
        ----------
            tickers (list): List of ticker symbols
            **kwargs: Additional arguments to pass to load_returns

        Returns
        -------
            pd.DataFrame: DataFrame of returns with tickers as columns

        """
        returns = {}
        for ticker in tickers:
            try:
                returns[ticker] = self.load_returns(ticker, **kwargs)
            except Exception as e:
                logger.warning(f"Skipping {ticker}: {str(e)}")
                continue

        return pd.DataFrame(returns)
