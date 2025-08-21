import argparse
import json
import warnings
from datetime import datetime
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import jarque_bera, shapiro
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from src.data.loader import DataLoader
from src.infra.logging import logger


# -------------------------------
# Helper for JSON serialization
# -------------------------------
def to_serializable(obj):
    """Convert numpy/pandas objects to JSON-serializable types"""
    if isinstance(obj, (np.generic,)):  # numpy scalar
        return obj.item()
    if isinstance(obj, (pd.Series, pd.Index)):  # pandas Series/Index
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):  # pandas DataFrame
        return obj.to_dict(orient="records")
    if pd.isna(obj):  # NaN → None
        return None
    return str(obj)  # fallback for anything else


# ------------------------
# Error metrics per quarter
# ------------------------
def compute_error_metrics(forecasts: pd.DataFrame, freq: str = "Q") -> pd.DataFrame:
    """
    Compute rolling error metrics (MAE, RMSE, Bias) per given period (default Quarterly).
    """
    forecasts = forecasts.copy()
    forecasts["date"] = pd.to_datetime(forecasts["date"])
    forecasts = forecasts.set_index("date")

    # Error calculations
    forecasts["error"] = forecasts["forecast"] - forecasts["actual"]
    forecasts["abs_error"] = forecasts["error"].abs()
    forecasts["squared_error"] = forecasts["error"] ** 2

    # Resample to period (Quarter)
    grouped = forecasts.resample(freq)
    metrics = grouped.apply(
        lambda g: pd.Series(
            {
                "MAE": g["abs_error"].mean(),
                "RMSE": np.sqrt(g["squared_error"].mean()),
                "Bias": g["error"].mean(),
                "Count": len(g),
            }
        )
    )
    return metrics


class GJRGARCHModel:
    """Production-ready GJR-GARCH model with enhanced diagnostics and stability"""

    def __init__(self, p: int = 1, q: int = 1, o: int = 1):
        self.p = p
        self.q = q
        self.o = o
        self.model = None
        self.results = None
        self.diagnostics = {}
        self.ticker = None
        self.window_size = None

    def fit(self, returns: pd.Series) -> None:
        """Fit GJR-GARCH model to returns with enhanced stability checks"""
        try:
            returns = returns.dropna()
            self.model = arch_model(
                returns * 100,  # Scale for better optimization
                mean="Zero",
                vol="GARCH",
                p=self.p,
                q=self.q,
                o=self.o,
                dist="StudentsT",
                rescale=False,
                power=2.0,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.results = self.model.fit(disp="off", update_freq=5)

            if not self._check_parameter_bounds():
                logger.warning("Parameter estimates outside recommended bounds")
            self._compute_diagnostics(returns)

            logger.info(
                "GJR-GARCH model fitted successfully",
                params=self.results.params.to_dict(),
                aic=self.results.aic,
                bic=self.results.bic,
            )
        except Exception as e:
            logger.error(f"Failed to fit model: {str(e)}", exc_info=True)
            raise

    def _check_parameter_bounds(self) -> bool:
        """Check if estimated parameters are within reasonable bounds"""
        if not self.results:
            return False

        params = self.results.params

        # Define reasonable bounds
        bounds = {"alpha[1]": (0, 0.3), "beta[1]": (0.5, 0.99), "gamma[1]": (-0.3, 0.3)}

        # Check each parameter
        all_within_bounds = True
        for param, (lower, upper) in bounds.items():
            if param in params:
                value = params[param]
                if value < lower or value > upper:
                    logger.warning(
                        f"Parameter {param}={value:.4f} outside recommended bounds ({lower}, {upper})"
                    )
                    all_within_bounds = False

        return all_within_bounds

    def _compute_diagnostics(self, returns: pd.Series) -> None:
        """Compute comprehensive model diagnostics and goodness-of-fit metrics"""
        try:
            residuals = self.results.resid / 100  # Scale back
            standardized_residuals = residuals / self.results.conditional_volatility

            # Normality tests
            jb_test = jarque_bera(standardized_residuals)
            shapiro_test = shapiro(standardized_residuals)

            # Autocorrelation tests
            lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)

            # Volatility clustering test
            squared_lb_test = acorr_ljungbox(residuals**2, lags=10, return_df=True)

            # Store diagnostics
            self.diagnostics = {
                "residuals_normality": {
                    "jarque_bera": {"statistic": jb_test[0], "pvalue": jb_test[1]},
                    "shapiro": {
                        "statistic": shapiro_test[0],
                        "pvalue": shapiro_test[1],
                    },
                },
                "residuals_autocorrelation": {
                    "ljung_box": lb_test.to_dict(),
                    "acf": dict(zip(range(11), acf(residuals, nlags=10))),
                },
                "volatility_clustering": {
                    "squared_ljung_box": squared_lb_test.to_dict()
                },
                "goodness_of_fit": {
                    "mse": mean_squared_error(returns[1:], self.forecast_insample()),
                    "r_squared": self._compute_r_squared(returns),
                },
                "stability_check": {
                    "stationary": self._check_stationarity(),
                    "explosive": self._check_explosive_volatility(),
                    "parameters_within_bounds": self._check_parameter_bounds(),
                },
            }
        except Exception as e:
            logger.warning(f"Could not compute all diagnostics: {str(e)}")
            self.diagnostics = {"error": str(e)}

    def _compute_r_squared(self, returns: pd.Series) -> float:
        """Compute pseudo R-squared for model fit with explicit mean calculation"""
        try:
            forecast = self.forecast_insample()
            mean_return = returns[1:].mean()  # Explicit mean calculation
            ss_res = np.sum((returns[1:] - forecast) ** 2)
            ss_tot = np.sum((returns[1:] - mean_return) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
        except Exception:
            return float("nan")

    def _check_stationarity(self) -> bool:
        """Check if the model parameters satisfy stationarity conditions"""
        if not self.results:
            return False

        alpha = self.results.params.get("alpha[1]", 0)
        beta = self.results.params.get("beta[1]", 0)
        gamma = self.results.params.get("gamma[1]", 0)

        # Stationarity condition for GJR-GARCH
        return (alpha + beta + 0.5 * gamma) < 1

    def _check_explosive_volatility(self) -> bool:
        """Check for explosive volatility patterns"""
        if not self.results:
            return True

        beta = self.results.params.get("beta[1]", 0)
        return (
            beta >= 0.99
        )  # Very high persistence may indicate near-explosive volatility

    def forecast(self, horizon: int = 1) -> Dict[str, float]:
        """Generate volatility forecast with error handling"""
        if not self.results:
            raise ValueError("Model not fitted. Call fit() first.")

        try:
            forecasts = self.results.forecast(horizon=horizon, reindex=False)
            variance = forecasts.variance.iloc[-1, 0] / 10000  # Scale back
            volatility = np.sqrt(variance)

            return {
                "sigma": volatility,
                "variance": variance,
                "date": pd.Timestamp.now(),  # Add timestamp for tracking
            }
        except Exception as e:
            logger.error(f"Forecast failed: {str(e)}")
            raise

    def forecast_insample(self) -> pd.Series:
        """Generate in-sample conditional volatility forecasts"""
        if not self.results:
            raise ValueError("Model not fitted. Call fit() first.")
        return pd.Series(
            self.results.conditional_volatility / 100,
            index=self.results.resid.index[1:],
        )

    def rolling_forecast(
        self, returns: pd.Series, window: int = 252, n_jobs: int = 1
    ) -> pd.DataFrame:
        """
        Generate rolling out-of-sample forecasts with progress tracking

        Parameters
        ----------
            returns (pd.Series): Time series of returns
            window (int): Rolling window size (default 252 trading days)
            n_jobs (int): Number of parallel jobs (default 1 for stability)

        Returns
        -------
            pd.DataFrame: Forecasts with actual values and dates

        """

        def _process_window(i):
            try:
                # Progress tracking
                if i % 10 == 0 or i == window:
                    progress = (i - window) / (len(returns) - window) * 100
                    logger.info(
                        f"Processing {progress:.1f}% complete ({i-window}/{len(returns)-window})"
                    )

                train_returns = returns.iloc[i - window : i]
                model = GJRGARCHModel(p=self.p, q=self.q, o=self.o)
                model.fit(train_returns)

                forecast = model.forecast(horizon=1)
                forecast["date"] = returns.index[i]
                forecast["actual"] = returns.iloc[i]
                forecast["actual_vol"] = np.abs(returns.iloc[i])
                return forecast
            except Exception as e:
                logger.warning(f"Skipping forecast at {returns.index[i]}: {str(e)}")
                return None

        try:
            # Parallel processing with progress tracking
            results = []
            for i in range(window, len(returns)):
                result = _process_window(i)
                if result is not None:
                    results.append(result)

            # Create DataFrame from successful forecasts
            forecasts = pd.DataFrame([r for r in results if r is not None])

            if not forecasts.empty:
                forecasts = forecasts.set_index("date")
                # Compute forecast errors
                forecasts["error"] = forecasts["actual_vol"] - forecasts["sigma"]
                forecasts["abs_error"] = np.abs(forecasts["error"])
                forecasts["squared_error"] = forecasts["error"] ** 2

                # Add rolling error metrics
                forecasts["rolling_mae"] = forecasts["abs_error"].rolling(21).mean()
                forecasts["rolling_rmse"] = np.sqrt(
                    forecasts["squared_error"].rolling(21).mean()
                )

            return forecasts

        except KeyboardInterrupt:
            logger.info("Rolling forecast interrupted by user - saving partial results")
            if "forecasts" in locals() and not forecasts.empty:
                return forecasts
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Rolling forecast failed: {str(e)}")
            raise

    def plot_forecasts(
        self, forecasts: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """Plot actual vs predicted volatility with error metrics"""
        if forecasts.empty:
            logger.warning("No forecasts to plot")
            return

        plt.figure(figsize=(14, 10))

        # Main plot - actual vs predicted volatility
        plt.subplot(2, 1, 1)
        plt.plot(
            forecasts.index,
            forecasts["actual_vol"],
            label="Actual Volatility",
            color="blue",
            alpha=0.6,
        )
        plt.plot(
            forecasts.index,
            forecasts["sigma"],
            label="Predicted Volatility",
            color="red",
            linestyle="--",
        )
        plt.title("Volatility Forecast vs Actual")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.legend()
        plt.grid(True)

        # Error metrics plot
        plt.subplot(2, 1, 2)
        if "rolling_mae" in forecasts.columns:
            plt.plot(
                forecasts.index,
                forecasts["rolling_mae"],
                label="21-Day Rolling MAE",
                color="green",
            )
        if "rolling_rmse" in forecasts.columns:
            plt.plot(
                forecasts.index,
                forecasts["rolling_rmse"],
                label="21-Day Rolling RMSE",
                color="purple",
            )
        plt.title("Forecast Error Metrics")
        plt.xlabel("Date")
        plt.ylabel("Error")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def save_results(self, forecasts: pd.DataFrame, path: str) -> None:
        """
        Save forecasts, diagnostics, and quarterly error metrics.

        Args:
        ----
            forecasts: DataFrame with rolling forecasts, actuals, and errors.
            path: Path to JSON file to write. A CSV with quarterly error metrics
                will be written in the same directory with suffix `_quarterly.csv`.

        """
        # Build JSON-friendly results dict
        results = {
            "metadata": {
                "ticker": self.ticker if self.ticker else "unknown",
                "model": f"GJR-GARCH({self.p},{self.q},{self.o})",
                "created_at": datetime.now().isoformat(),
                "window_size": self.window_size if self.window_size else "unknown",
            },
            "parameters": self.results.params.to_dict()
            if self.results is not None
            else {},
            "diagnostics": self.diagnostics,
            "forecasts": forecasts.reset_index().to_dict(orient="records")
            if not forecasts.empty
            else [],
            "metrics": {
                "mae": float(forecasts["abs_error"].mean())
                if not forecasts.empty
                else None,
                "rmse": float(np.sqrt((forecasts["error"] ** 2).mean()))
                if not forecasts.empty
                else None,
                "mse": float((forecasts["error"] ** 2).mean())
                if not forecasts.empty
                else None,
                "last_rolling_mae": float(forecasts["rolling_mae"].iloc[-1])
                if not forecasts.empty and "rolling_mae" in forecasts.columns
                else None,
                "last_rolling_rmse": float(forecasts["rolling_rmse"].iloc[-1])
                if not forecasts.empty and "rolling_rmse" in forecasts.columns
                else None,
            },
        }

        # Derive output paths
        out_json = path
        out_csv = path.replace(".json", "_quarterly.csv")

        # Compute and save quarterly error metrics
        if not forecasts.empty:
            try:
                metrics_quarterly = compute_error_metrics(
                    forecasts.reset_index(),  # ensure 'date' column exists
                    freq="Q",
                )
                metrics_quarterly.to_csv(out_csv, index=True)
                print(f"[INFO] Saved quarterly error metrics → {out_csv}")
            except Exception as e:
                logger.warning(f"Could not save quarterly error metrics: {e}")

        # Save JSON
        with open(out_json, "w") as f:
            json.dump(results, f, indent=4, default=to_serializable)
        print(f"[INFO] Saved forecasts → {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced GJR-GARCH Model")
    parser.add_argument("--ticker", type=str, default="SPY", help="Stock ticker symbol")
    parser.add_argument("--window", type=int, default=252, help="Rolling window size")
    parser.add_argument("--p", type=int, default=1, help="GARCH order")
    parser.add_argument("--q", type=int, default=1, help="ARCH order")
    parser.add_argument("--o", type=int, default=1, help="Asymmetric term order")
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (1 for reliability)",
    )
    parser.add_argument("--save", type=str, default=None, help="Path to save results")
    args = parser.parse_args()

    try:
        logger.info(f"Loading data for {args.ticker}")
        loader = DataLoader()
        returns = loader.load_returns(args.ticker)

        logger.info(f"Initializing GJR-GARCH({args.p},{args.q},{args.o}) model")
        model = GJRGARCHModel(p=args.p, q=args.q, o=args.o)
        model.ticker = args.ticker
        model.window_size = args.window

        logger.info(f"Starting rolling forecasts with window={args.window}")
        forecasts = model.rolling_forecast(
            returns, window=args.window, n_jobs=args.n_jobs
        )

        if model.results:
            print("\nModel Parameters:")
            print(model.results.params.to_markdown())

            print("\nModel Diagnostics:")
            print(pd.DataFrame(model.diagnostics).to_markdown())

        if not forecasts.empty:
            print("\nForecast Metrics:")
            metrics = {
                "MAE": forecasts["abs_error"].mean(),
                "RMSE": np.sqrt((forecasts["error"] ** 2).mean()),
                "MSE": (forecasts["error"] ** 2).mean(),
                "Last_21D_MAE": forecasts["rolling_mae"].iloc[-1]
                if "rolling_mae" in forecasts.columns
                else float("nan"),
                "Last_21D_RMSE": forecasts["rolling_rmse"].iloc[-1]
                if "rolling_rmse" in forecasts.columns
                else float("nan"),
            }
            print(pd.Series(metrics).to_markdown(floatfmt=".6f"))

            print("\nRecent Forecasts:")
            print(forecasts.tail().to_markdown(floatfmt=".4f"))

            model.plot_forecasts(forecasts)

        if args.save:
            model.save_results(forecasts, args.save)
            logger.info(f"Results saved to {args.save}")

    except Exception as e:
        logger.error(f"Failed to run model: {str(e)}", exc_info=True)
        raise
