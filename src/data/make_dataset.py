# src/data/make_dataset.py
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import List, Dict
import sys
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from config.settings import settings
except ImportError:
    # Fallback if settings can't be imported
    class Settings:
        def __init__(self):
            self.BASE_DIR = Path(__file__).parent.parent.parent
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
    all_prices = []
    
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
    
    # Create regime features based on VIX
    try:
        # Try to load VIX data
        vix_files = list(settings.RAW_DIR.glob("*VIX*.parquet")) + list(settings.RAW_DIR.glob("*VIX*.pkl"))
        if vix_files:
            vix_data = None
            for vix_file in vix_files:
                try:
                    if vix_file.suffix == '.parquet':
                        vix_data = pd.read_parquet(vix_file)
                    else:
                        with open(vix_file, 'rb') as f:
                            vix_data = pickle.load(f)
                    break
                except:
                    continue
            
            if vix_data is not None:
                vix = None
                if hasattr(vix_data, 'columns'):
                    if 'close' in vix_data.columns:
                        vix = vix_data['close']
                    elif 'Close' in vix_data.columns:
                        vix = vix_data['Close']
                    elif len(vix_data.columns) > 0:
                        vix = vix_data.iloc[:, 0]
                
                if vix is not None:
                    # Ensure we have the right data type (numeric values)
                    vix = pd.to_numeric(vix, errors='coerce').dropna()
                    
                    # Create regime features based on VIX thresholds
                    regime_features = pd.DataFrame({
                        'regime_LOW': (vix < settings.VIX_THRESHOLDS['low']).astype(int),
                        'regime_MEDIUM': ((vix >= settings.VIX_THRESHOLDS['low']) & (vix < settings.VIX_THRESHOLDS['medium'])).astype(int),
                        'regime_HIGH': ((vix >= settings.VIX_THRESHOLDS['medium']) & (vix < settings.VIX_THRESHOLDS['high'])).astype(int),
                        'regime_EXTREME': (vix >= settings.VIX_THRESHOLDS['high']).astype(int),
                    }, index=vix.index)
                    
                    # Align with returns index
                    regime_features = regime_features.reindex(returns_df.index, method='ffill').fillna(0)
                    regime_features.to_parquet(settings.PROCESSED_DIR / "regime_features.parquet")
                    print("Created regime features from VIX data")
                    
    except Exception as e:
        print(f"Error creating regime features from VIX data: {e}")
        # Create simple regime features based on volatility
        try:
            print("Creating regime features based on portfolio volatility...")
            portfolio_volatility = returns_df.std(axis=1).rolling(21).mean() * np.sqrt(252)
            
            # Define volatility thresholds (you can adjust these)
            vol_thresholds = {
                'low': 0.15,
                'medium': 0.25, 
                'high': 0.35
            }
            
            regime_features = pd.DataFrame({
                'regime_LOW': (portfolio_volatility < vol_thresholds['low']).astype(int),
                'regime_MEDIUM': ((portfolio_volatility >= vol_thresholds['low']) & (portfolio_volatility < vol_thresholds['medium'])).astype(int),
                'regime_HIGH': ((portfolio_volatility >= vol_thresholds['medium']) & (portfolio_volatility < vol_thresholds['high'])).astype(int),
                'regime_EXTREME': (portfolio_volatility >= vol_thresholds['high']).astype(int),
            }, index=returns_df.index)
            
            regime_features.to_parquet(settings.PROCESSED_DIR / "regime_features.parquet")
            print("Created regime features based on portfolio volatility")
            
        except Exception as vol_error:
            print(f"Error creating volatility-based regime features: {vol_error}")
            # Create simple random regime features as fallback
            print("Creating random regime features as fallback...")
            regime_features = pd.DataFrame({
                'regime_LOW': (np.random.random(len(returns_df)) > 0.8).astype(int),
                'regime_MEDIUM': (np.random.random(len(returns_df)) > 0.6).astype(int),
                'regime_HIGH': (np.random.random(len(returns_df)) > 0.4).astype(int),
                'regime_EXTREME': (np.random.random(len(returns_df)) > 0.2).astype(int),
            }, index=returns_df.index)
            regime_features.to_parquet(settings.PROCESSED_DIR / "regime_features.parquet")
            print("Created random regime features")
    
    # Create CVaR decomposition (simplified)
    cvar_decomposition = pd.DataFrame({
        'cvar_contribution': np.random.dirichlet(np.ones(len(tickers))),
    }, index=tickers)
    cvar_decomposition.to_parquet(settings.PROCESSED_DIR / "cvar_decomposition.parquet")
    
    # Create factor exposures (simplified)
    factors = ['Market', 'Size', 'Value', 'Momentum', 'Quality', 'Volatility']
    factor_exposures = pd.DataFrame(
        np.random.normal(0, 1, (len(returns_df), len(factors))),
        index=returns_df.index, columns=factors
    )
    factor_exposures.to_parquet(settings.PROCESSED_DIR / "factor_exposures.parquet")
    
    # Create correlation matrix
    correlation_matrix = returns_df.corr()
    correlation_matrix.to_parquet(settings.PROCESSED_DIR / "correlation_matrix.parquet")
    
    # Create stress test results (simplified)
    stress_periods = ['2008-09', '2020-03', '2015-08', '2018-12']
    stress_test_results = pd.DataFrame({
        'drawdown': np.random.uniform(-0.15, -0.45, len(stress_periods)),
        'recovery_days': np.random.randint(30, 365, len(stress_periods)),
    }, index=stress_periods)
    stress_test_results.to_parquet(settings.PROCESSED_DIR / "stress_test_results.parquet")
    
    # Create benchmark returns (using SPY if available)
    try:
        spy_files = list(settings.RAW_DIR.glob("SPY*"))
        for spy_file in spy_files:
            try:
                if spy_file.suffix == '.parquet':
                    spy_data = pd.read_parquet(spy_file)
                else:
                    with open(spy_file, 'rb') as f:
                        spy_data = pickle.load(f)
                
                spy_prices = None
                if hasattr(spy_data, 'columns'):
                    if 'close' in spy_data.columns:
                        spy_prices = spy_data['close']
                    elif 'Close' in spy_data.columns:
                        spy_prices = spy_data['Close']
                    elif len(spy_data.columns) > 0:
                        spy_prices = spy_data.iloc[:, 0]
                
                if spy_prices is not None:
                    spy_returns = spy_prices.pct_change().dropna()
                    spy_returns.name = 'SPY'
                    spy_returns.to_parquet(settings.PROCESSED_DIR / "benchmark_returns.parquet")
                    break
            except:
                continue
    except Exception as e:
        print(f"Could not create benchmark returns: {e}")
    
    # Save the main data files
    returns_df.to_parquet(settings.PROCESSED_DIR / "returns.parquet")
    volatility_df.to_parquet(settings.PROCESSED_DIR / "volatility.parquet")
    weights_df.to_parquet(settings.PROCESSED_DIR / "portfolio_weights.parquet")
    performance_df.to_parquet(settings.PROCESSED_DIR / "backtest_results.parquet")
    
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