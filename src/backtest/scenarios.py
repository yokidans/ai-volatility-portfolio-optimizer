from dataclasses import dataclass
from enum import Enum
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import multivariate_t

# Use a default seed if settings is not available
try:
    from src.config import settings

    DEFAULT_SEED = settings.SEED
except (ImportError, AttributeError):
    DEFAULT_SEED = 42
    print(f"Using default seed: {DEFAULT_SEED}")


class CrisisPeriod(Enum):
    """Major financial crisis periods"""

    GFC_2008 = ("2007-12-01", "2009-06-30")
    EURO_2011 = ("2010-04-01", "2012-03-31")
    COVID_2020 = ("2020-02-01", "2020-04-30")
    INFLATION_2022 = ("2022-01-01", "2022-12-31")


@dataclass
class StressScenario:
    name: str
    start_date: str
    end_date: str
    volatility_multiplier: float
    correlation_shift: float
    liquidity_impact: float
    description: str


class StressScenarios:
    """Comprehensive stress testing framework for portfolio backtesting"""

    def __init__(self, seed: int = DEFAULT_SEED):  # Use DEFAULT_SEED here
        self.rng = np.random.default_rng(seed)
        self.crisis_periods = {
            "gfc_2008": StressScenario(
                name="Global Financial Crisis 2008",
                start_date="2007-12-01",
                end_date="2009-06-30",
                volatility_multiplier=2.8,
                correlation_shift=0.6,
                liquidity_impact=0.4,
                description="Lehman Brothers collapse, credit freeze",
            ),
            "covid_2020": StressScenario(
                name="COVID-19 Crash",
                start_date="2020-02-20",
                end_date="2020-03-23",
                volatility_multiplier=3.2,
                correlation_shift=0.7,
                liquidity_impact=0.6,
                description="Global pandemic market crash",
            ),
            "inflation_2022": StressScenario(
                name="Inflation Shock 2022",
                start_date="2022-01-01",
                end_date="2022-12-31",
                volatility_multiplier=2.2,
                correlation_shift=0.5,
                liquidity_impact=0.3,
                description="Aggressive Fed tightening cycle",
            ),
        }

    def apply_crisis_scenario(
        self, prices: pd.DataFrame, crisis_name: str
    ) -> pd.DataFrame:
        """Apply historical crisis parameters to price data"""
        if crisis_name not in self.crisis_periods:
            raise ValueError(
                f"Unknown crisis: {crisis_name}. Available: {list(self.crisis_periods.keys())}"
            )

        scenario = self.crisis_periods[crisis_name]
        crisis_prices = prices.copy()

        # Identify crisis period in data
        mask = (crisis_prices.index >= scenario.start_date) & (
            crisis_prices.index <= scenario.end_date
        )

        if not mask.any():
            logger.warning(
                f"No data found for crisis period {scenario.start_date} to {scenario.end_date}"
            )
            return crisis_prices

        # Apply volatility shock
        crisis_returns = crisis_prices.loc[mask].pct_change().dropna()
        shocked_returns = crisis_returns * scenario.volatility_multiplier

        # Apply correlation breakdown (assets move together)
        if len(crisis_returns.columns) > 1:
            base_corr = crisis_returns.corr().values
            shocked_corr = self._increase_correlation(
                base_corr, scenario.correlation_shift
            )
            shocked_returns = self._apply_correlation_structure(
                shocked_returns, shocked_corr
            )

        # Reconstruct prices from shocked returns
        crisis_start_price = crisis_prices.loc[mask].iloc[0]
        shocked_prices = crisis_start_price * (1 + shocked_returns).cumprod()

        crisis_prices.loc[mask] = shocked_prices
        crisis_prices = crisis_prices.ffill()

        logger.info(
            f"Applied {scenario.name} scenario: {scenario.volatility_multiplier}x vol, "
            f"{scenario.correlation_shift} corr shift"
        )

        return crisis_prices

    def monte_carlo_stress_test(
        self,
        returns: pd.DataFrame,
        n_simulations: int = 1000,
        stress_intensity: float = 2.0,
    ) -> Dict[str, np.ndarray]:
        """Generate Monte Carlo stress scenarios with fat tails"""
        n_assets = returns.shape[1]
        n_periods = returns.shape[0]

        # Fit multivariate t-distribution (fat tails)
        mu = returns.mean().values
        cov = returns.cov().values
        dof = 4.0  # Degrees of freedom for fat tails

        # Generate stressed returns
        stressed_returns = np.zeros((n_simulations, n_periods, n_assets))

        for i in range(n_simulations):
            # Multivariate t-distribution with stressed covariance
            stressed_cov = cov * stress_intensity
            t_samples = multivariate_t.rvs(
                mean=mu,
                shape=stressed_cov,
                df=dof,
                size=n_periods,
                random_state=self.rng,
            )
            stressed_returns[i] = t_samples

        # Calculate portfolio metrics for each simulation
        portfolio_returns = stressed_returns.mean(axis=2)  # Equal-weighted
        max_drawdowns = np.array(
            [self._calculate_max_drawdown(pr) for pr in portfolio_returns]
        )
        final_values = (1 + portfolio_returns).prod(axis=1)

        return {
            "stressed_returns": stressed_returns,
            "max_drawdowns": max_drawdowns,
            "final_values": final_values,
            "var_95": np.percentile(final_values, 5),
            "cvar_95": final_values[
                final_values <= np.percentile(final_values, 5)
            ].mean(),
        }

    def liquidity_shock_scenario(
        self, prices: pd.DataFrame, shock_magnitude: float = 0.3
    ) -> pd.DataFrame:
        """Simulate liquidity crisis with widened bid-ask spreads"""
        shocked_prices = prices.copy()

        # Calculate normal bid-ask spread (assume 5bps for liquid ETFs)
        normal_spread = 0.0005

        for asset in shocked_prices.columns:
            # Apply liquidity shock: widened spreads + price impact
            spread_shock = normal_spread * (1 + shock_magnitude)

            # Simulate price impact of large orders
            returns = shocked_prices[asset].pct_change()
            liquidity_impact = returns.abs() * shock_magnitude

            # Apply to prices (asymmetric: sells impact more than buys)
            shocked_returns = returns.copy()
            negative_returns = returns < 0
            shocked_returns[negative_returns] = returns[negative_returns] * (
                1 + liquidity_impact[negative_returns]
            )

            # Reconstruct prices
            shocked_prices[asset] = (
                shocked_prices[asset].iloc[0] * (1 + shocked_returns).cumprod()
            )

        return shocked_prices

    def flash_crash_scenario(
        self,
        prices: pd.DataFrame,
        crash_duration: int = 30,  # minutes
        recovery_days: int = 3,
    ) -> pd.DataFrame:
        """Simulate a flash crash scenario"""
        shocked_prices = prices.copy()

        # Find a random date for the flash crash
        crash_start = self.rng.choice(prices.index[:-crash_duration])
        crash_end = prices.index[prices.index.get_loc(crash_start) + crash_duration]

        # Apply immediate drop (20-30%)
        crash_severity = self.rng.uniform(0.2, 0.3)
        shocked_prices.loc[crash_start:crash_end] *= 1 - crash_severity

        # Gradual recovery
        recovery_mask = (shocked_prices.index > crash_end) & (
            shocked_prices.index
            <= prices.index[prices.index.get_loc(crash_end) + recovery_days]
        )
        recovery_prices = shocked_prices.loc[recovery_mask]

        if not recovery_prices.empty:
            # Linear recovery to pre-crash levels
            recovery_factor = np.linspace(0, 1, len(recovery_prices))
            pre_crash_level = shocked_prices.loc[crash_start]
            current_level = shocked_prices.loc[crash_end]

            recovery_adjustment = (pre_crash_level - current_level) * recovery_factor
            shocked_prices.loc[recovery_mask] += recovery_adjustment.values

        logger.info(
            f"Simulated flash crash: {crash_severity:.1%} drop on {crash_start}, "
            f"{recovery_days} day recovery"
        )

        return shocked_prices

    def _increase_correlation(
        self, corr_matrix: np.ndarray, shift_intensity: float
    ) -> np.ndarray:
        """Increase correlation matrix towards 1 during crises"""
        n_assets = corr_matrix.shape[0]
        shifted_corr = corr_matrix.copy()

        # Move all correlations towards +1
        shifted_corr = shifted_corr + (1 - shifted_corr) * shift_intensity

        # Ensure diagonal remains 1 and matrix is PSD
        np.fill_diagonal(shifted_corr, 1.0)
        shifted_corr = self._make_psd(shifted_corr)

        return shifted_corr

    def _apply_correlation_structure(
        self, returns: pd.DataFrame, target_corr: np.ndarray
    ) -> pd.DataFrame:
        """Apply target correlation structure to returns"""
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(target_corr)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(self._make_psd(target_corr))

        # Transform returns to have target correlation
        uncorrelated_returns = returns.values @ np.linalg.inv(
            np.linalg.cholesky(returns.corr())
        )
        correlated_returns = uncorrelated_returns @ L

        return pd.DataFrame(
            correlated_returns, index=returns.index, columns=returns.columns
        )

    @staticmethod
    def _make_psd(matrix: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """Make matrix positive semi-definite"""
        w, V = np.linalg.eigh(matrix)
        w = np.maximum(w, epsilon)
        return V @ np.diag(w) @ V.T

    @staticmethod
    def _calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown from return series"""
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        return np.max(drawdown) if len(drawdown) > 0 else 0.0

    def generate_stress_report(
        self, backtest_results: Dict, baseline_results: Dict
    ) -> pd.DataFrame:
        """Generate comprehensive stress test report"""
        report_data = []

        for crisis_name, scenario in self.crisis_periods.items():
            crisis_perf = backtest_results.get(crisis_name, {})
            baseline_perf = baseline_results.get("baseline", {})

            if crisis_perf and baseline_perf:
                report_data.append(
                    {
                        "Scenario": scenario.name,
                        "Period": f"{scenario.start_date} to {scenario.end_date}",
                        "Max Drawdown": crisis_perf.get("max_drawdown", 0),
                        "Baseline DD": baseline_perf.get("max_drawdown", 0),
                        "DD Increase": crisis_perf.get("max_drawdown", 0)
                        - baseline_perf.get("max_drawdown", 0),
                        "Sharpe Ratio": crisis_perf.get("sharpe_ratio", 0),
                        "Baseline Sharpe": baseline_perf.get("sharpe_ratio", 0),
                        "CVaR (5%)": crisis_perf.get("cvar_5", 0),
                        "Baseline CVaR": baseline_perf.get("cvar_5", 0),
                    }
                )

        return pd.DataFrame(report_data)


# Example usage
if __name__ == "__main__":
    # Load sample data
    from src.data.fetchers import DataFetcher

    fetcher = DataFetcher()
    prices, _ = fetcher.fetch_all_data()
    prices = prices.pivot(index="date", columns="ticker", values="close")

    # Initialize stress tester
    stress_tester = StressScenarios()

    # Test historical crises
    for crisis in ["gfc_2008", "covid_2020", "inflation_2022"]:
        try:
            stressed_prices = stress_tester.apply_crisis_scenario(prices, crisis)
            print(f"Applied {crisis} scenario successfully")
        except Exception as e:
            print(f"Failed to apply {crisis}: {str(e)}")

    # Run Monte Carlo stress test
    returns = prices.pct_change().dropna()
    mc_results = stress_tester.monte_carlo_stress_test(returns, n_simulations=1000)

    print(f"MC Stress Test - 95% VaR: {mc_results['var_95']:.2%}")
    print(f"MC Stress Test - 95% CVaR: {mc_results['cvar_95']:.2%}")
