import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.backtest.scenarios import StressScenarios


@pytest.fixture
def mock_portfolio_weights():
    return pd.DataFrame(
        {"SPY": [0.4], "TSLA": [0.3], "BND": [0.3]},
        index=[pd.to_datetime("2020-02-20")],
    )


@pytest.fixture
def mock_prices():
    dates = pd.date_range(start="2020-02-10", periods=20)
    return pd.DataFrame(
        {
            "SPY": [330 + i * 0.5 for i in range(20)],
            "TSLA": [180 + i * 1.2 for i in range(20)],
            "BND": [82 + i * 0.05 for i in range(20)],
        },
        index=dates,
    )


def test_covid_crash(mock_prices, mock_portfolio_weights):
    """Test portfolio behavior during COVID crash."""
    scenarios = StressScenarios()
    covid_prices = scenarios.apply_covid_crash(mock_prices.copy())

    # Prices should drop significantly
    assert covid_prices["SPY"].iloc[-1] < mock_prices["SPY"].iloc[-1] * 0.8
    assert covid_prices["TSLA"].iloc[-1] < mock_prices["TSLA"].iloc[-1] * 0.7

    # Run backtest
    backtester = BacktestEngine()
    results = backtester.run_backtest(covid_prices, mock_portfolio_weights)

    # Portfolio should lose value but less than worst asset
    max_drawdown = (
        results["portfolio_value"].min() - results["portfolio_value"].iloc[0]
    ) / results["portfolio_value"].iloc[0]
    assert max_drawdown < -0.15  # Significant drawdown
    assert max_drawdown > -0.5  # But not as bad as worst asset
