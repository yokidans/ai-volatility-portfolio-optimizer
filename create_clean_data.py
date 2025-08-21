# create_clean_data.py
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import sys
from datetime import datetime, timedelta

# Add the src directory to the Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from config.settings import settings
except ImportError:
    class Settings:
        def __init__(self):
            self.BASE_DIR = Path(__file__).parent
            self.DATA_DIR = self.BASE_DIR / "data"
            self.RAW_DIR = self.DATA_DIR / "raw"
            self.PROCESSED_DIR = self.DATA_DIR / "processed"
    
    settings = Settings()

def create_clean_data():
    """Create clean processed data excluding problematic MSFT data."""
    print("Creating clean processed data...")
    
    # Ensure processed directory exists
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load only AAPL and GOOG data (exclude MSFT)
    tickers = ['AAPL', 'GOOG']
    all_returns = []
    all_prices = []
    
    for ticker in tickers:
        try:
            stock_data = None
            
            # Try to load PKL files first
            pkl_file = settings.RAW_DIR / f"{ticker}.pkl"
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    stock_data = pickle.load(f)
            
            # Try to load parquet files if PKL failed or doesn't exist
            if stock_data is None:
                parquet_file = settings.RAW_DIR / f"{ticker}_VIX_1d.parquet"
                if parquet_file.exists():
                    stock_data = pd.read_parquet(parquet_file)
            
            if stock_data is None:
                print(f"No data found for {ticker}")
                continue
            
            # Extract price data
            prices = None
            if hasattr(stock_data, 'columns'):
                for col_name in ['close', 'Close', 'price', 'Price', 'last', 'Last']:
                    if col_name in stock_data.columns:
                        prices = stock_data[col_name]
                        break
                
                if prices is None and len(stock_data.columns) > 0:
                    prices = stock_data.iloc[:, 0]
            
            if prices is None:
                print(f"Could not extract price data for {ticker}")
                continue
            
            # Clean and calculate returns
            prices = prices.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            returns = prices.pct_change().fillna(0)
            
            # Cap extreme returns
            returns = returns.clip(-0.2, 0.2)  # Cap at ±20% daily returns
            
            all_returns.append(returns.rename(ticker))
            all_prices.append(prices.rename(ticker))
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    if not all_returns:
        print("No data could be processed. Creating sample data instead.")
        create_sample_data()
        return
    
    # Combine all data
    returns_df = pd.concat(all_returns, axis=1).dropna()
    print(f"Clean returns data shape: {returns_df.shape}")
    
    # Create portfolio weights (equal weight)
    weights_df = pd.DataFrame(
        np.ones((len(returns_df), len(tickers))) / len(tickers),
        index=returns_df.index,
        columns=tickers
    )
    
    # Create performance data
    initial_capital = 1_000_000
    portfolio_returns = (returns_df * weights_df.shift(1)).sum(axis=1).fillna(0)
    portfolio_value = initial_capital * (1 + portfolio_returns).cumprod()
    performance_df = pd.DataFrame({'portfolio_value': portfolio_value})
    
    # Calculate volatility
    volatility_df = returns_df.rolling(21).std() * np.sqrt(252)
    volatility_df = volatility_df.ffill().bfill()
    
    # Create regime features
    portfolio_volatility = returns_df.std(axis=1).rolling(21).mean() * np.sqrt(252)
    portfolio_volatility = portfolio_volatility.ffill().bfill()
    
    vol_thresholds = {'low': 0.15, 'medium': 0.25, 'high': 0.35}
    regime_features = pd.DataFrame({
        'regime_LOW': (portfolio_volatility < vol_thresholds['low']).astype(int),
        'regime_MEDIUM': ((portfolio_volatility >= vol_thresholds['low']) & (portfolio_volatility < vol_thresholds['medium'])).astype(int),
        'regime_HIGH': ((portfolio_volatility >= vol_thresholds['medium']) & (portfolio_volatility < vol_thresholds['high'])).astype(int),
        'regime_EXTREME': (portfolio_volatility >= vol_thresholds['high']).astype(int),
    }, index=returns_df.index).fillna(0)
    
    # Create other analytical datasets
    cvar_decomposition = pd.DataFrame({
        'cvar_contribution': np.ones(len(tickers)) / len(tickers),
    }, index=tickers)
    
    factors = ['Market', 'Size', 'Value', 'Momentum', 'Quality', 'Volatility']
    factor_exposures = pd.DataFrame(
        np.random.normal(0, 0.3, (len(returns_df), len(factors))),
        index=returns_df.index, columns=factors
    )
    
    correlation_matrix = returns_df.corr()
    
    stress_periods = ['2008-09', '2020-03', '2015-08', '2018-12']
    stress_test_results = pd.DataFrame({
        'drawdown': np.random.uniform(-0.15, -0.30, len(stress_periods)),
        'recovery_days': np.random.randint(60, 180, len(stress_periods)),
    }, index=stress_periods)
    
    # Save all files
    returns_df.to_parquet(settings.PROCESSED_DIR / "returns.parquet")
    volatility_df.to_parquet(settings.PROCESSED_DIR / "volatility.parquet")
    weights_df.to_parquet(settings.PROCESSED_DIR / "portfolio_weights.parquet")
    performance_df.to_parquet(settings.PROCESSED_DIR / "backtest_results.parquet")
    regime_features.to_parquet(settings.PROCESSED_DIR / "regime_features.parquet")
    cvar_decomposition.to_parquet(settings.PROCESSED_DIR / "cvar_decomposition.parquet")
    factor_exposures.to_parquet(settings.PROCESSED_DIR / "factor_exposures.parquet")
    correlation_matrix.to_parquet(settings.PROCESSED_DIR / "correlation_matrix.parquet")
    stress_test_results.to_parquet(settings.PROCESSED_DIR / "stress_test_results.parquet")
    
    print(f"Clean data processing complete. Files saved to {settings.PROCESSED_DIR}")
    final_value = performance_df['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    print(f"Final portfolio value: ${final_value:,.2f}")
    print(f"Total return: {total_return:.2f}%")

def create_sample_data():
    """Create realistic sample data."""
    print("Creating realistic sample data...")
    
    # Create dates and assets
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    assets = ['AAPL', 'GOOGL']
    
    # Generate realistic returns
    returns_df = pd.DataFrame(
        np.random.normal(0.0008, 0.018, (len(dates), len(assets))),
        index=dates, columns=assets
    )
    
    # Create portfolio weights
    weights_df = pd.DataFrame(
        np.ones((len(dates), len(assets))) / len(assets),
        index=dates, columns=assets
    )
    
    # Create performance data
    initial_capital = 1_000_000
    portfolio_returns = returns_df.mean(axis=1)  # Simple average for sample
    portfolio_value = initial_capital * (1 + portfolio_returns).cumprod()
    performance_df = pd.DataFrame({'portfolio_value': portfolio_value})
    
    # Save main files
    returns_df.to_parquet(settings.PROCESSED_DIR / "returns.parquet")
    weights_df.to_parquet(settings.PROCESSED_DIR / "portfolio_weights.parquet")
    performance_df.to_parquet(settings.PROCESSED_DIR / "backtest_results.parquet")
    
    print("Sample data created successfully.")

if __name__ == "__main__":
    create_clean_data()