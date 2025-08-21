# debug_data.py
import pandas as pd
import numpy as np
from pathlib import Path
import sys

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
            self.PROCESSED_DIR = self.DATA_DIR / "processed"
    
    settings = Settings()

def debug_data():
    """Debug the processed data to identify issues."""
    print("Debugging processed data...")
    
    # Check if processed files exist
    processed_files = list(settings.PROCESSED_DIR.glob("*.parquet"))
    print(f"Found {len(processed_files)} processed files")
    
    # Load and inspect returns data
    returns_file = settings.PROCESSED_DIR / "returns.parquet"
    if returns_file.exists():
        returns_df = pd.read_parquet(returns_file)
        print(f"\nReturns data shape: {returns_df.shape}")
        print(f"Returns data columns: {returns_df.columns.tolist()}")
        print(f"Date range: {returns_df.index.min()} to {returns_df.index.max()}")
        print(f"Number of NaN values: {returns_df.isna().sum().sum()}")
        
        # Show basic statistics
        print("\nReturns statistics:")
        print(returns_df.describe())
        
        # Check for extreme values
        print(f"\nExtreme values check:")
        for col in returns_df.columns:
            max_val = returns_df[col].max()
            min_val = returns_df[col].min()
            if abs(max_val) > 1 or abs(min_val) > 1:  # Returns should be small decimals
                print(f"  {col}: min={min_val:.6f}, max={max_val:.6f} (WARNING: Extreme values!)")
            else:
                print(f"  {col}: min={min_val:.6f}, max={max_val:.6f}")
    
    # Load and inspect performance data
    performance_file = settings.PROCESSED_DIR / "backtest_results.parquet"
    if performance_file.exists():
        performance_df = pd.read_parquet(performance_file)
        print(f"\nPerformance data shape: {performance_df.shape}")
        print(f"Performance data columns: {performance_df.columns.tolist()}")
        print(f"Portfolio value range: {performance_df['portfolio_value'].min():.2f} to {performance_df['portfolio_value'].max():.2f}")
        
        # Check if portfolio value is collapsing
        initial_value = performance_df['portfolio_value'].iloc[0]
        final_value = performance_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        print(f"Actual total return: {total_return:.2%}")
        
        if final_value < initial_value * 0.1:  # Lost more than 90%
            print("WARNING: Portfolio value has collapsed!")

if __name__ == "__main__":
    debug_data()