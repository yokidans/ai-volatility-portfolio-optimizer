import numpy as np
import pandas as pd

from src.config import settings
from src.infra.logging import logger


class DataValidator:
    """Lightweight validator without pandera dependency"""

    def __init__(self):
        self.anomaly_threshold = 4.0

    def validate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Manual validation of returns data"""
        try:
            # Type conversion
            df["date"] = pd.to_datetime(df["date"])
            df["close"] = pd.to_numeric(df["close"])
            df["returns"] = pd.to_numeric(df["returns"])

            # Value validation
            valid_tickers = settings.TICKERS + settings.BENCHMARKS
            if not df["ticker"].isin(valid_tickers).all():
                invalid = set(df["ticker"]) - set(valid_tickers)
                raise ValueError(f"Invalid tickers: {invalid}")

            if (df["close"] < 0).any():
                raise ValueError("Negative close prices found")

            if ((df["returns"] < -1) | (df["returns"] > 5)).any():
                raise ValueError("Returns outside valid range [-1, 5]")

            # Anomaly detection
            returns = df["returns"].dropna()
            z_scores = (returns - returns.mean()) / returns.std()
            anomalies = np.abs(z_scores) > self.anomaly_threshold

            if anomalies.any():
                logger.warning(f"Anomalies detected: {anomalies.sum()}")

            return df
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise
