import numpy as np
import pandas as pd
from typing import Dict, Literal
from enum import Enum
from src.config import settings
from src.infra.logging import logger

class VolatilityRegime(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EXTREME = 4

class RegimeDetector:
    """Advanced volatility regime detection with adaptive thresholds and hysteresis."""
    
    def __init__(self):
        self.thresholds = settings.VIX_THRESHOLDS
        self.hysteresis = 0.9  # Prevents rapid switching between regimes
        self.current_regime = None
        
    def detect_regimes(self, vix_series: pd.Series) -> pd.Series:
        """Detect volatility regimes with adaptive thresholds and memory."""
        regimes = pd.Series(index=vix_series.index, dtype='object')
        
        # Initial state
        if vix_series.iloc[0] < self.thresholds["low"]:
            self.current_regime = VolatilityRegime.LOW
        elif vix_series.iloc[0] < self.thresholds["medium"]:
            self.current_regime = VolatilityRegime.MEDIUM
        elif vix_series.iloc[0] < self.thresholds["high"]:
            self.current_regime = VolatilityRegime.HIGH
        else:
            self.current_regime = VolatilityRegime.EXTREME
        
        regimes.iloc[0] = self.current_regime
        
        # Apply hysteresis-based switching
        for i in range(1, len(vix_series)):
            vix = vix_series.iloc[i]
            
            if self.current_regime == VolatilityRegime.LOW:
                if vix > self.thresholds["low"] * (1 + self.hysteresis):
                    self.current_regime = VolatilityRegime.MEDIUM
            elif self.current_regime == VolatilityRegime.MEDIUM:
                if vix > self.thresholds["medium"] * (1 + self.hysteresis):
                    self.current_regime = VolatilityRegime.HIGH
                elif vix < self.thresholds["low"] * (1 - self.hysteresis):
                    self.current_regime = VolatilityRegime.LOW
            elif self.current_regime == VolatilityRegime.HIGH:
                if vix > self.thresholds["high"] * (1 + self.hysteresis):
                    self.current_regime = VolatilityRegime.EXTREME
                elif vix < self.thresholds["medium"] * (1 - self.hysteresis):
                    self.current_regime = VolatilityRegime.MEDIUM
            else:  # EXTREME
                if vix < self.thresholds["high"] * (1 - self.hysteresis):
                    self.current_regime = VolatilityRegime.HIGH
            
            regimes.iloc[i] = self.current_regime
        
        logger.info("Volatility regimes detected", 
                   regime_counts=regimes.value_counts().to_dict())
        return regimes
    
    def create_regime_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Create regime-aware features including transition indicators."""
        vix = price_data['VIX'].copy()
        regimes = self.detect_regimes(vix)
        
        # One-hot encode regimes
        regime_dummies = pd.get_dummies(regimes, prefix='regime')
        
        # Create transition indicators
        shifted_regimes = regimes.shift(1)
        transitions = (regimes != shifted_regimes) & (~shifted_regimes.isna())
        transition_dummies = pd.get_dummies(regimes[transitions], prefix='transition')
        
        # Combine features
        features = pd.concat([regime_dummies, transition_dummies], axis=1)
        features = features.fillna(0).astype(int)
        
        # Add VIX level features
        features['vix_level'] = vix
        features['vix_change'] = vix.pct_change()
        features['vix_ma_ratio'] = vix / vix.rolling(21).mean()
        
        return features