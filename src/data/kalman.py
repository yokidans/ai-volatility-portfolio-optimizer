import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pykalman import KalmanFilter
from src.infra.logging import logger

class KalmanImputer:
    """Robust missing data imputation using Kalman Filters with adaptive parameters."""
    
    def __init__(self, n_dim: int = 1):
        self.n_dim = n_dim
        self.kf = None
    
    def fit(self, observed_data: pd.Series) -> None:
        """Fit Kalman Filter parameters using EM algorithm."""
        observed_data = observed_data.copy().interpolate(limit=5)
        observed_values = observed_data.values.reshape(-1, self.n_dim)
        
        # Initialize with heuristic values for financial time series
        initial_state_mean = np.nanmean(observed_values, axis=0)
        initial_state_covariance = np.eye(self.n_dim) * np.nanvar(observed_values, axis=0)
        
        transition_matrix = np.eye(self.n_dim)  # Random walk
        observation_matrix = np.eye(self.n_dim)
        
        self.kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            em_vars=['transition_covariance', 'observation_covariance']
        )
        
        # EM algorithm to learn parameters
        self.kf = self.kf.em(observed_values, n_iter=10)
        logger.info("Kalman Filter trained", 
                   transition_covariance=self.kf.transition_covariance,
                   observation_covariance=self.kf.observation_covariance)
    
    def transform(self, observed_data: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Apply Kalman smoothing and return imputed series with uncertainty."""
        if self.kf is None:
            raise ValueError("Kalman Filter not fitted. Call fit() first.")
            
        observed_values = observed_data.copy().interpolate(limit=5).values.reshape(-1, self.n_dim)
        state_means, state_covariances = self.kf.smooth(observed_values)
        
        imputed_series = pd.Series(
            state_means.flatten(),
            index=observed_data.index,
            name=f"{observed_data.name}_imputed"
        )
        
        uncertainty = pd.Series(
            np.sqrt(state_covariances.flatten()),
            index=observed_data.index,
            name=f"{observed_data.name}_uncertainty"
        )
        
        return imputed_series, uncertainty