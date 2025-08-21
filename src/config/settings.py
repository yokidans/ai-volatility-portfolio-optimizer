# src/config/settings.py
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # Paths
    DATA_DIR: Path = Path("data")
    RAW_DIR: Path = DATA_DIR / "raw"
    INTERIM_DIR: Path = DATA_DIR / "interim"
    PROCESSED_DIR: Path = DATA_DIR / "processed"

    # Seeds
    GLOBAL_SEED: int = 42

    # VIX thresholds (✅ added here)
    VIX_THRESHOLDS: dict = field(
        default_factory=lambda: {"low": 15, "medium": 25, "high": 30, "extreme": 40}
    )

    # Portfolio constraints
    MAX_WEIGHT: float = 0.3
    MIN_WEIGHT: float = 0.01
    TURNOVER_CAP: float = 0.2
    CASH_SLEEVE: float = 0.05
    LEVERAGE_LIMIT: float = 1.0

    # Risk
    CVAR_ALPHA: float = 0.05
    VAR_ALPHA: float = 0.05

    # Backtesting
    INITIAL_CAPITAL: float = 1_000_000.0
    COMMISSION_RATE: float = 0.0005
    SLIPPAGE_RATE: float = 0.0002
    REBALANCE_FREQUENCY: str = "monthly"

    # API keys
    FRED_API_KEY: str = os.getenv("FRED_API_KEY")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")


# ✅ instantiate once for use everywhere
settings = Settings()
