"""
src/data/fetchers.py

Data fetching utilities for equities, ETFs, and macro series.
Designed for reproducibility: all calls have retry, timeout, and deterministic saving.

Author: GMF Quant Lab
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_yfinance_data(
    tickers: Union[str, List[str]],
    start: str = "2015-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    cache: bool = True,
    retries: int = 3,
    pause: int = 2,
) -> pd.DataFrame:
    """
    Fetch historical price data from Yahoo Finance.

    Parameters
    ----------
    tickers : str or list of str
        One or multiple ticker symbols (e.g., "SPY", ["SPY", "^VIX"]).
    start : str
        Start date (YYYY-MM-DD).
    end : str, optional
        End date (YYYY-MM-DD). Defaults to today if None.
    interval : str
        Data frequency ("1d", "1wk", "1mo").
    cache : bool
        If True, caches results to `data/raw/<tickers>.parquet`.
    retries : int
        Number of retry attempts if API call fails.
    pause : int
        Seconds to wait between retries.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with columns per ticker (Adj Close).

    """
    if isinstance(tickers, str):
        tickers = [tickers]

    # cache path
    fname = f"{'_'.join([t.replace('^','') for t in tickers])}_{interval}.parquet"
    cache_path = DATA_DIR / fname

    if cache and cache_path.exists():
        logger.info(f"Loading cached data: {cache_path}")
        return pd.read_parquet(cache_path)

    # fetch with retry
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Fetching data from yfinance for {tickers}...")
            df = yf.download(
                tickers,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            break
        except Exception as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(pause)
            else:
                raise

    # Ensure proper format: keep Adj Close only
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Adj Close"].copy()
    elif "Adj Close" in df.columns:
        df = df[["Adj Close"]].copy()
        df = df.rename(columns={"Adj Close": tickers[0]})
    else:
        raise ValueError("Unexpected yfinance DataFrame format.")

    # Save cache
    if cache:
        df.to_parquet(cache_path)
        logger.info(f"Saved data to {cache_path}")

    return df


# Example usage when run standalone
if __name__ == "__main__":
    data = get_yfinance_data(["SPY", "^VIX"], start="2020-01-01", end="2023-01-01")
    print(data.head())
