from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BacktestEngine:
    """High-performance vectorized backtesting engine with realistic market frictions."""

    def __init__(
        self,
        commission: float = 0.0005,  # 5 bps per trade
        slippage: float = 0.0002,  # 2 bps slippage
        delay: int = 1,  # execution delay (days)
    ):
        self.commission = commission
        self.slippage = slippage
        self.delay = delay

    def run_backtest(
        self, prices: pd.DataFrame, signals: pd.DataFrame, initial_capital: float = 1e6
    ) -> Dict[str, pd.DataFrame]:
        """
        Run vectorized backtest with transaction costs and execution delay.

        Args:
        ----
            prices: DataFrame of asset prices (rows=dates, cols=assets).
            signals: DataFrame of portfolio weights (same shape as prices).
            initial_capital: starting cash.

        Returns:
        -------
            Dictionary of portfolio metrics.

        """
        # Align dates
        aligned_idx = prices.index.intersection(signals.index)
        prices = prices.loc[aligned_idx]
        signals = signals.loc[aligned_idx]

        # Execution delay: shift signals
        exec_signals = signals.shift(self.delay).fillna(0.0)

        # Normalize weights row-wise (avoid leverage blowup)
        exec_signals = exec_signals.div(exec_signals.abs().sum(axis=1), axis=0).fillna(
            0.0
        )

        # Compute portfolio values
        n_days, n_assets = prices.shape
        portfolio_value = pd.Series(index=prices.index, dtype=float)
        cash = pd.Series(index=prices.index, dtype=float)
        commissions = pd.Series(index=prices.index, dtype=float)
        turnover = pd.Series(index=prices.index, dtype=float)

        # First allocation
        portfolio_value.iloc[0] = initial_capital
        cash.iloc[0] = initial_capital
        positions = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for t in range(1, n_days):
            date = prices.index[t]
            prev_date = prices.index[t - 1]

            # Target portfolio value split across assets
            target_value = exec_signals.loc[date] * portfolio_value.iloc[t - 1]
            target_positions = target_value / prices.loc[date]

            # Trades
            trades = target_positions - positions.loc[prev_date]
            trade_costs = (
                trades.abs() * prices.loc[date] * (self.commission + self.slippage)
            ).sum()

            # Update positions + cash
            positions.loc[date] = target_positions
            cash.loc[date] = cash.loc[prev_date] - trade_costs

            # Portfolio value
            portfolio_value.loc[date] = (
                positions.loc[date] * prices.loc[date]
            ).sum() + cash.loc[date]

            commissions.loc[date] = trade_costs
            turnover.loc[date] = (
                trades.abs().sum() / portfolio_value.loc[date]
                if portfolio_value.loc[date] > 0
                else 0
            )

        returns = portfolio_value.pct_change().fillna(0)

        # --- Performance metrics ---
        total_return = portfolio_value.iloc[-1] / initial_capital - 1
        sharpe = (
            np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
        )
        max_dd = ((portfolio_value / portfolio_value.cummax()) - 1).min()

        print("=== Backtest Summary ===")
        print(f"Final Portfolio Value: {portfolio_value.iloc[-1]:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Sharpe: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        print(f"Total Commissions Paid: {commissions.sum():,.2f}")

        # --- Visualization ---
        plt.figure(figsize=(12, 6))
        portfolio_value.plot(label="Portfolio Value")
        (initial_capital * (1 + returns.cumsum())).plot(
            alpha=0.5, linestyle="--", label="Cumulative Return"
        )
        plt.title("Equity Curve")
        plt.legend()
        plt.show()

        return {
            "portfolio_value": portfolio_value,
            "returns": returns,
            "positions": positions,
            "cash": cash,
            "turnover": turnover,
            "commissions": commissions,
        }

    def stress_test(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        stress_periods: Dict[str, Tuple[str, str]],
        initial_capital: float = 1e6,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Run backtest on specific stress periods (dict of {name: (start_date, end_date)}).
        """
        results = {}
        for name, (start, end) in stress_periods.items():
            sub_prices = prices.loc[start:end]
            sub_signals = signals.loc[start:end]
            print(f"\n--- Stress Test: {name} ---")
            results[name] = self.run_backtest(sub_prices, sub_signals, initial_capital)
        return results


# === Example usage ===
if __name__ == "__main__":
    # Fake example data (2 assets, 100 days)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    prices = pd.DataFrame(
        {
            "AssetA": np.cumprod(1 + 0.001 * np.random.randn(100)),
            "AssetB": np.cumprod(1 + 0.0012 * np.random.randn(100)),
        },
        index=dates,
    )

    # Example signals: random weights
    signals = pd.DataFrame(
        np.random.randn(100, 2), index=dates, columns=["AssetA", "AssetB"]
    )

    engine = BacktestEngine()
    results = engine.run_backtest(prices, signals, initial_capital=1e6)
