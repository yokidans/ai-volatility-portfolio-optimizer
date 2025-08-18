import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from src.config import settings
from src.infra.logging import logger

class BacktestEngine:
    """High-performance vectorized backtesting engine with realistic market frictions."""
    
    def __init__(self, 
                commission: float = 0.0005,  # 5 bps per trade
                slippage: float = 0.0002,   # 2 bps slippage
                delay: int = 1):            # 1-day execution delay
        self.commission = commission
        self.slippage = slippage
        self.delay = delay
        
    def run_backtest(self,
                    prices: pd.DataFrame,
                    signals: pd.DataFrame,
                    initial_capital: float = 1e6) -> Dict[str, pd.DataFrame]:
        """Run vectorized backtest with transaction costs and execution delay."""
        # Align dates and ensure proper indexing
        aligned_idx = prices.index.intersection(signals.index)
        prices = prices.loc[aligned_idx]
        signals = signals.loc[aligned_idx]
        
        # Initialize portfolio
        n_days = len(prices)
        n_assets = len(prices.columns)
        positions = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        cash = pd.Series(initial_capital, index=prices.index)
        portfolio_value = pd.Series(0.0, index=prices.index)
        turnover = pd.Series(0.0, index=prices.index)
        commissions = pd.Series(0.0, index=prices.index)
        
        # Vectorized backtest loop
        for t in range(self.delay, n_days):
            # Current date and execution date (with delay)
            current_date = prices.index[t]
            execution_date = prices.index[t-self.delay]
            
            # Calculate target positions based on signals
            target_weights = signals.loc[execution_date]
            target_values = target_weights * portfolio_value[t-1] if t > 0 else target_weights * initial_capital
            target_positions = target_values / prices.loc[current_date]
            
            # Calculate trades and apply transaction costs
            trades = target_positions - positions.iloc[t-1] if t > 0 else target_positions
            trade_costs = trades.abs() * prices.loc[current_date] * (self.commission + self.slippage)
            
            # Update positions and cash
            positions.loc[current_date] = target_positions
            cash.loc[current_date] = cash.iloc[t-1] - trade_costs.sum() if t > 0 else initial_capital - trade_costs.sum()
            
            # Record metrics
            portfolio_value.loc[current_date] = (positions.loc[current_date] * prices.loc[current_date]).sum() + cash.loc[current_date]
            turnover.loc[current_date] = trades.abs().sum() / portfolio_value.loc[current_date]
            commissions.loc[current_date] = trade_costs.sum()
        
        # Calculate returns and performance metrics
        returns = portfolio_value.pct_change().fillna(0)
        
        logger.info("Backtest completed",
                   final_value=portfolio_value.iloc[-1],
                   total_return=portfolio_value.iloc[-1]/initial_capital - 1,
                   avg_turnover=turnover.mean(),
                   total_commissions=commissions.sum())
        
        return {
            'portfolio_value': portfolio_value,
            'returns': returns,
            'positions': positions,
            'cash': cash,
            'turnover': turnover,
            'commissions': commissions
        }
    
    def stress_test(self, 
                   prices: pd.DataFrame,
                   signals: pd.DataFrame,
                   stress_periods: Dict[str, Tuple[str, str]],
                   initial_capital: float = 1e6) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Run backtest on specific stress periods."""
        results = {}
        
        for name, (start, end) in stress_periods.items():
            period_prices = prices.loc[start:end]
            period_signals = signals.loc[start:end]
            
            results[name] = self.run_backtest(
                prices=period_prices,
                signals=period_signals,
                initial_capital=initial_capital
            )
            
            logger.info(f"Stress test completed for {name}",
                       period_return=results[name]['portfolio_value'].iloc[-1]/initial_capital - 1)
        
        return results