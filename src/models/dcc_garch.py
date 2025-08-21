import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.linalg import sqrtm


class DCCGARCH:
    """Robust DCC-GARCH implementation with proper model initialization"""

    def __init__(
        self,
        univariate_garch_params: Optional[Dict] = None,
        dcc_params: Optional[Dict] = None,
    ):
        self.univariate_models = {}
        self.dcc_model = None
        self.residuals = None
        self.correlations = None

        # Default parameters
        self.univariate_garch_params = univariate_garch_params or {
            "mean": "Zero",
            "vol": "GARCH",
            "p": 1,
            "q": 1,
            "dist": "normal",
            "rescale": True,
        }

        self.dcc_params = dcc_params or {
            "a": 0.05,  # DCC alpha
            "b": 0.90,  # DCC beta
        }

    def fit(self, returns: pd.DataFrame) -> None:
        """Fit the DCC-GARCH model"""
        # Stage 1: Fit univariate GARCH models
        self._fit_univariate_models(returns)

        # Stage 2: Fit DCC model
        self._fit_dcc_model()

        # Compute conditional correlations
        self._compute_conditional_correlations()

        return self

    def _fit_univariate_models(self, returns: pd.DataFrame) -> None:
        """Fit univariate GARCH models"""
        self.univariate_models = {}
        standardized_residuals = pd.DataFrame(index=returns.index)

        for asset in returns.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = arch_model(returns[asset], **self.univariate_garch_params)
                results = model.fit(disp="off")
                self.univariate_models[asset] = results
                standardized_residuals[asset] = (
                    results.resid / results.conditional_volatility
                )

        self.residuals = standardized_residuals

    def _fit_dcc_model(self) -> None:
        """Fit DCC model using the standardized residuals"""
        if self.residuals is None:
            raise ValueError("Univariate models not fitted")

        # Calculate the unconditional correlation matrix
        n_assets = len(self.univariate_models)
        Qbar = self.residuals.cov().values

        # Initialize Qt matrices
        n_obs = len(self.residuals)
        Qt = np.zeros((n_obs + 1, n_assets, n_assets))
        Qt[0] = Qbar

        # Get DCC parameters
        a = self.dcc_params["a"]
        b = self.dcc_params["b"]

        # DCC recursion
        residuals_array = self.residuals.values
        for t in range(1, n_obs + 1):
            Qt[t] = (
                Qbar * (1 - a - b)
                + a * np.outer(residuals_array[t - 1], residuals_array[t - 1])
                + b * Qt[t - 1]
            )

        # Store the Qt matrices (excluding the initial Qbar)
        self.Qt = Qt[1:]

    def _compute_conditional_correlations(self) -> None:
        """Compute dynamic conditional correlations from Qt matrices"""
        n_assets = len(self.univariate_models)
        n_obs = len(self.residuals)
        self.correlations = np.zeros((n_obs, n_assets, n_assets))

        for t in range(n_obs):
            Qt_t = self.Qt[t]
            diag_Qt = np.diag(np.diag(Qt_t))
            sqrt_Qt_t = sqrtm(diag_Qt)
            inv_sqrt_Qt_t = np.linalg.inv(sqrt_Qt_t)
            R_t = inv_sqrt_Qt_t @ Qt_t @ inv_sqrt_Qt_t

            # Ensure valid correlation matrix
            R_t = (R_t + R_t.T) / 2  # Force symmetry
            np.fill_diagonal(R_t, 1)  # Ensure unit diagonal
            self.correlations[t] = R_t

    def forecast_correlation(self, horizon: int = 1) -> np.ndarray:
        """Forecast correlation matrix"""
        if not hasattr(self, "Qt") or self.Qt is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Forecast univariate volatilities
        univariate_forecasts = {}
        for asset, model in self.univariate_models.items():
            forecast = model.forecast(horizon=horizon, reindex=False)
            univariate_forecasts[asset] = np.sqrt(forecast.variance.iloc[-1, 0])

        # DCC forecast (simple persistence)
        a = self.dcc_params["a"]
        b = self.dcc_params["b"]
        Q_forecast = np.mean(self.Qt, axis=0)  # Long-run average

        # Compute forecasted correlation matrix
        diag_Q = np.diag(np.diag(Q_forecast))
        sqrt_Q = sqrtm(diag_Q)
        inv_sqrt_Q = np.linalg.inv(sqrt_Q)
        R_forecast = inv_sqrt_Q @ Q_forecast @ inv_sqrt_Q

        # Combine with univariate forecasts
        D = np.diag(list(univariate_forecasts.values()))
        cov_forecast = D @ R_forecast @ D

        return cov_forecast
