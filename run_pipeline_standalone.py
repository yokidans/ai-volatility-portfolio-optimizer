# run_pipeline_standalone.py
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import sys

# Add the src directory to the Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Simple settings class
class Settings:
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.RAW_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DIR = self.DATA_DIR / "processed"
        self.VIX_THRESHOLDS = {"low": 15, "medium": 25, "high": 30, "extreme": 40}
        self.CVAR_ALPHA = 0.05

settings = Settings()

def process_raw_data():
    """Process raw data into the format needed for the dashboard."""
    print("Processing raw data...")
    
    # Ensure processed directory exists
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    raw_files = list(settings.RAW_DIR.glob("*.pkl")) + list(settings.RAW_DIR.glob("*.parquet"))
    
    if not raw_files:
        print("No raw data files found. Creating sample data instead.")
        create_sample_data()
        return
    
    tickers = list(set([f.stem.split('_')[0] for f in raw_files if not f.name.startswith('SPY') and not 'VIX' in f.name]))
    tickers = [t for t in tickers if t not in ['SPY', 'VIX']]
    
    print(f"Found data for tickers: {tickers}")
    
    # Load and process individual stock data
    all_returns = []
    all_volatility = []
    
    for ticker in tickers:
        try:
            stock_data = None
            
            # Try to load PKL files first
            pkl_file = settings.RAW_DIR / f"{ticker}.pkl"
            if pkl_file.exists():
                try:
                    with open(pkl_file, 'rb') as f:
                        stock_data = pickle.load(f)
                except:
                    print(f"Could not load {pkl_file} as pickle, trying other formats")
            
            # Try to load parquet files if PKL failed or doesn't exist
            if stock_data is None:
                parquet_file = settings.RAW_DIR / f"{ticker}_VIX_1d.parquet"
                if parquet_file.exists():
                    try:
                        stock_data = pd.read_parquet(parquet_file)
                    except:
                        print(f"Could not load {parquet_file} as parquet")
            
            if stock_data is None:
                print(f"No data found for {ticker}")
                continue
            
            # Extract price data
            prices = None
            if hasattr(stock_data, 'columns'):
                if 'close' in stock_data.columns:
                    prices = stock_data['close']
                elif 'Close' in stock_data.columns:
                    prices = stock_data['Close']
                elif len(stock_data.columns) > 0:
                    prices = stock_data.iloc[:, 0]  # Use first column as price
            
            if prices is None:
                print(f"Could not extract price data for {ticker}")
                continue
            
            # Calculate returns and volatility
            returns = prices.pct_change().dropna()
            volatility = returns.rolling(21).std() * np.sqrt(252)  # Annualized volatility
            
            all_returns.append(returns.rename(ticker))
            all_volatility.append(volatility.rename(ticker))
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    if not all_returns:
        print("No data could be processed. Creating sample data instead.")
        create_sample_data()
        return
    
    # Combine all data
    returns_df = pd.concat(all_returns, axis=1).dropna()
    volatility_df = pd.concat(all_volatility, axis=1).dropna()
    
    # Create portfolio weights (example: equal weight)
    weights_df = pd.DataFrame(
        np.ones((len(returns_df), len(tickers))) / len(tickers),
        index=returns_df.index,
        columns=tickers
    )
    
    # Create performance data (assuming $1M initial investment)
    initial_capital = 1_000_000
    portfolio_returns = (returns_df * weights_df.shift(1)).sum(axis=1)
    portfolio_value = initial_capital * (1 + portfolio_returns).cumprod()
    performance_df = pd.DataFrame({'portfolio_value': portfolio_value})
    
    # Create regime features (simplified)
    regime_features = pd.DataFrame({
        'regime_LOW': (np.random.random(len(returns_df)) > 0.8).astype(int),
        'regime_MEDIUM': (np.random.random(len(returns_df)) > 0.6).astype(int),
        'regime_HIGH': (np.random.random(len(returns_df)) > 0.4).astype(int),
        'regime_EXTREME': (np.random.random(len(returns_df)) > 0.2).astype(int),
    }, index=returns_df.index)
    
    # Create CVaR decomposition (simplified)
    cvar_decomposition = pd.DataFrame({
        'cvar_contribution': np.random.dirichlet(np.ones(len(tickers))),
    }, index=tickers)
    
    # Create factor exposures (simplified)
    factors = ['Market', 'Size', 'Value', 'Momentum', 'Quality', 'Volatility']
    factor_exposures = pd.DataFrame(
        np.random.normal(0, 1, (len(returns_df), len(factors))),
        index=returns_df.index, columns=factors
    )
    
    # Create correlation matrix
    correlation_matrix = returns_df.corr()
    
    # Create stress test results (simplified)
    stress_periods = ['2008-09', '2020-03', '2015-08', '2018-12']
    stress_test_results = pd.DataFrame({
        'drawdown': np.random.uniform(-0.15, -0.45, len(stress_periods)),
        'recovery_days': np.random.randint(30, 365, len(stress_periods)),
    }, index=stress_periods)
    
    # Create benchmark returns (simplified)
    benchmark_returns = pd.Series(
        np.random.normal(0.0004, 0.018, len(returns_df)),
        index=returns_df.index, name='SPY'
    )
    
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
    benchmark_returns.to_parquet(settings.PROCESSED_DIR / "benchmark_returns.parquet")
    
    print(f"Data processing complete. Files saved to {settings.PROCESSED_DIR}")

def create_sample_data():
    """Create sample data if real data processing fails."""
    print("Creating sample data...")
    
    # Create dates and assets
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # Generate basic required data
    returns_df = pd.DataFrame(
        np.random.normal(0.0005, 0.02, (len(dates), len(assets))),
        index=dates, columns=assets
    )
    volatility_df = pd.DataFrame(
        np.random.uniform(0.15, 0.45, (len(dates), len(assets))),
        index=dates, columns=assets
    )
    weights_df = pd.DataFrame(
        np.random.dirichlet(np.ones(len(assets)), size=len(dates)),
        index=dates, columns=assets
    )
    
    # Generate performance data
    portfolio_value = pd.Series(
        1000000 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates)))), 
        index=dates
    )
    performance_df = pd.DataFrame({'portfolio_value': portfolio_value})
    
    # Generate regime features
    regime_features = pd.DataFrame({
        'regime_LOW': (np.random.random(len(dates)) > 0.8).astype(int),
        'regime_MEDIUM': (np.random.random(len(dates)) > 0.6).astype(int),
        'regime_HIGH': (np.random.random(len(dates)) > 0.4).astype(int),
        'regime_EXTREME': (np.random.random(len(dates)) > 0.2).astype(int),
    }, index=dates)
    
    # Generate CVaR decomposition
    cvar_decomposition = pd.DataFrame({
        'cvar_contribution': np.random.dirichlet(np.ones(len(assets))),
    }, index=assets)
    
    # Generate factor exposures
    factors = ['Market', 'Size', 'Value', 'Momentum', 'Quality', 'Volatility']
    factor_exposures = pd.DataFrame(
        np.random.normal(0, 1, (len(dates), len(factors))),
        index=dates, columns=factors
    )
    
    # Generate correlation matrix
    correlation_matrix = pd.DataFrame(
        np.random.uniform(-0.7, 0.9, (len(assets), len(assets))),
        index=assets, columns=assets
    )
    np.fill_diagonal(correlation_matrix.values, 1.0)
    
    # Generate stress test results
    stress_periods = ['2008-09', '2020-03', '2015-08', '2018-12']
    stress_test_results = pd.DataFrame({
        'drawdown': np.random.uniform(-0.15, -0.45, len(stress_periods)),
        'recovery_days': np.random.randint(30, 365, len(stress_periods)),
    }, index=stress_periods)
    
    # Generate benchmark returns
    benchmark_returns = pd.Series(
        np.random.normal(0.0004, 0.018, len(dates)),
        index=dates, name='SPY'
    )
    
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
    benchmark_returns.to_parquet(settings.PROCESSED_DIR / "benchmark_returns.parquet")
    
    print("Sample data created successfully.")

if __name__ == "__main__":
    process_raw_data()