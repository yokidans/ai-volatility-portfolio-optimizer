import numpy as np
import pandas as pd
from src.config import settings
from src.infra.logging import logger
from src.data.validate import DataValidator
from arch import arch_model
from scipy.signal import periodogram
from statsmodels.tsa.stattools import adfuller

class FeatureBuilder:
    """Production-ready feature engineering with all fixes"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.vol_windows = [5, 21, 63]  # Weekly, monthly, quarterly
        self.min_periods = 5  # Reduced from 15 to avoid window conflicts
        
    def build_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Final optimized feature pipeline"""
        try:
            # Validation and initialization
            prices = self.validator.validate_returns(prices)
            features = pd.DataFrame(index=prices.index)
            
            # 1. Core returns features
            features = self._create_returns_features(features, prices)
            
            # 2. Enhanced volatility features
            features = self._create_volatility_features(features, prices)
            
            # 3. Robust frequency features
            features = self._create_frequency_features(features, prices)
            
            # 4. Derived features
            features = self._create_derived_features(features)
            
            return self._final_processing(features)
            
        except Exception as e:
            logger.error(f"Feature building failed: {str(e)}")
            raise
    
    def _create_returns_features(self, features, prices):
        """Core returns transformations"""
        features['returns'] = prices['returns']
        features['abs_returns'] = prices['returns'].abs()
        features['sign_returns'] = np.sign(prices['returns'])
        features['scaled_returns'] = prices['returns'] * 100  # For volatility models
        return features
    
    def _create_volatility_features(self, features, prices):
        """Professional volatility features"""
        # GARCH modeling with fallback
        try:
            am = arch_model(
                features['scaled_returns'],
                vol='Garch',
                p=1,
                q=1,
                dist='normal',  # Changed from skewt to avoid convergence issues
                rescale=False
            )
            res = am.fit(disp='off', show_warning=False)
            features['garch_vol'] = res.conditional_volatility / 100
        except Exception as e:
            logger.warning(f"Using EWMA fallback: {str(e)}")
            features['garch_vol'] = (
                features['scaled_returns']
                .ewm(span=21, min_periods=5)
                .std()
                .div(100)
            )
        
        # Rolling volatility metrics
        for window in self.vol_windows:
            min_p = min(5, window)  # Ensure min_periods <= window
            features[f'vol_{window}d'] = (
                features['scaled_returns']
                .rolling(window, min_periods=min_p)
                .std()
                .div(100)
            )
            features[f'range_{window}d'] = (
                (prices['high'].rolling(window).max() - 
                 prices['low'].rolling(window).min())
                .div(prices['close'].rolling(window).mean())
            )
        
        return features
    
    def _create_frequency_features(self, features, prices):
        """Robust frequency domain features"""
        returns = prices['returns'].dropna()
        
        if len(returns) > 30:  # Sufficient for meaningful analysis
            try:
                freq, power = periodogram(
                    returns - returns.mean(),
                    detrend='linear',
                    scaling='spectrum'
                )
                
                # Business-relevant frequency bands
                weekly_band = (freq >= 1/6) & (freq <= 1/4)
                monthly_band = (freq >= 1/22) & (freq <= 1/20)
                
                # Align with existing index
                valid_idx = features.index.intersection(prices.index[:-30])
                
                features.loc[valid_idx, 'freq_power_weekly'] = np.sum(power[weekly_band])
                features.loc[valid_idx, 'freq_power_monthly'] = np.sum(power[monthly_band])
                
            except Exception as e:
                logger.warning(f"Frequency analysis skipped: {str(e)}")
        
        return features
    
    def _create_derived_features(self, features):
        """Advanced feature combinations"""
        # Volatility regime features
        features['vol_ratio_short_long'] = (
            features['vol_5d'] / features['vol_63d'].replace(0, np.nan)
        )
        features['vol_regime'] = (
            (features['vol_5d'] > features['vol_21d'])
            .astype(int)
        )
        
        # Log transforms with clipping
        features['log_vol'] = np.log(features['garch_vol'].clip(lower=1e-6))
        
        return features
    
    def _final_processing(self, features):
        """Production-ready data quality checks"""
        # Handle infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill time-series features
        ts_features = ['garch_vol', 'vol_5d', 'vol_21d', 'vol_63d']
        features[ts_features] = features[ts_features].ffill()
        
        # Drop any remaining NA values
        features = features.dropna()
        
        # Final formatting
        return features.round(6)
    
    @staticmethod
    def prepare_test_data(days=500):
        """Generate institutional-quality test data"""
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
        
        # Volatility regimes
        regime = np.zeros(days)
        regime[250:] = 1  # High vol regime in second half
        
        # Regime-dependent volatility
        vol = np.where(regime == 0, 
                      np.random.uniform(0.01, 0.02, days),
                      np.random.uniform(0.03, 0.05, days))
        
        # Generate autocorrelated returns
        returns = np.zeros(days)
        for i in range(1, days):
            returns[i] = 0.1 * returns[i-1] + np.random.normal(0, vol[i])
        
        # Simulate realistic OHLC data
        close = 150 * np.cumprod(1 + returns)
        open = close * (1 + np.random.normal(0, 0.002, days))
        high = close * (1 + np.abs(np.random.normal(0, 0.005, days)))
        low = close * (1 - np.abs(np.random.normal(0, 0.005, days)))
        
        return pd.DataFrame({
            'date': dates,
            'ticker': ['AAPL'] * days,
            'open': open,
            'high': high,
            'low': low,
            'close': close,
            'returns': returns,
            'volume': np.random.lognormal(14, 0.5, days).astype(int)
        })
