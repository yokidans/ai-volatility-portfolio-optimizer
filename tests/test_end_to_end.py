import pytest

from src.backtest.engine import BacktestEngine
from src.config import settings
from src.data.build_features import FeatureBuilder
from src.data.fetchers import DataFetcher
from src.models.dcc_garch import DCCGARCH
from src.models.gjr_garch import GJRGARCHModel
from src.portfolio.cvar_opt import CVaROptimizer


@pytest.mark.integration
def test_full_pipeline():
    """Test the entire pipeline from data fetching to backtesting."""
    # 1. Fetch data
    fetcher = DataFetcher()
    assets, macro = fetcher.fetch_all_data()

    # 2. Build features
    feature_builder = FeatureBuilder()
    features, returns = feature_builder.build_features(assets, macro)
    assert not features.empty
    assert not returns.empty

    # 3. Train volatility models
    garch_model = GJRGARCHModel()
    garch_model.fit(returns["SPY"])
    forecasts = garch_model.rolling_forecast(returns["SPY"])
    assert len(forecasts) == len(returns) - settings.ROLLING_WINDOW

    # 4. Train DCC-GARCH
    dcc_model = DCCGARCH()
    asset_returns = assets.pivot(index="date", columns="ticker", values="returns")[
        settings.TICKERS
    ]
    dcc_model.fit(asset_returns.dropna())
    corr_forecast = dcc_model.forecast_correlation()
    assert corr_forecast.shape == (len(settings.TICKERS), len(settings.TICKERS))

    # 5. Portfolio optimization
    optimizer = CVaROptimizer()
    result = optimizer.optimize(asset_returns.dropna())
    assert result is not None
    assert len(result["weights"]) == len(settings.TICKERS)

    # 6. Backtesting
    weights = pd.DataFrame(
        [result["weights"]], columns=settings.TICKERS, index=[asset_returns.index[-1]]
    )
    backtester = BacktestEngine()
    prices = assets.pivot(index="date", columns="ticker", values="close")[
        settings.TICKERS
    ]
    backtest_results = backtester.run_backtest(prices, weights)
    assert not backtest_results["portfolio_value"].empty
