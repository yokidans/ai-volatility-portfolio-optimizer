from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class VolatilityVisualizer:
    """Enhanced visualization tools for volatility modeling."""

    @staticmethod
    def plot_volatility_clusters(
        returns: pd.Series, window: int = 21, save_path: Optional[str] = None
    ) -> None:
        """
        Plot returns with volatility clusters highlighted.

        Parameters
        ----------
            returns (pd.Series): Time series of returns
            window (int): Rolling window for volatility calculation
            save_path (str): Optional path to save the figure

        """
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)

        # Create figure
        plt.figure(figsize=(14, 8))

        # Plot returns
        plt.plot(returns.index, returns, color="gray", alpha=0.4, label="Daily Returns")

        # Highlight high volatility periods
        high_vol = rolling_vol > rolling_vol.quantile(0.9)
        plt.scatter(
            returns.index[high_vol],
            returns[high_vol],
            color="red",
            alpha=0.6,
            label="High Volatility Days",
        )

        # Add rolling volatility
        plt.plot(
            rolling_vol.index,
            rolling_vol,
            color="blue",
            linewidth=1.5,
            label=f"{window}-Day Rolling Volatility",
        )

        plt.title("Volatility Clusters in Returns")
        plt.xlabel("Date")
        plt.ylabel("Returns / Volatility")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_residuals_diagnostics(
        residuals: pd.Series, save_path: Optional[str] = None
    ) -> None:
        """
        Plot diagnostic plots for model residuals.

        Parameters
        ----------
            residuals (pd.Series): Model residuals
            save_path (str): Optional path to save the figure

        """
        plt.figure(figsize=(14, 10))

        # Time series plot
        plt.subplot(2, 2, 1)
        plt.plot(residuals.index, residuals)
        plt.title("Residuals Over Time")
        plt.xlabel("Date")
        plt.ylabel("Residual")
        plt.grid(True)

        # Histogram with KDE
        plt.subplot(2, 2, 2)
        sns.histplot(residuals, kde=True, stat="density")
        plt.title("Residuals Distribution")
        plt.xlabel("Residual")
        plt.grid(True)

        # Q-Q plot
        plt.subplot(2, 2, 3)
        from statsmodels.graphics.gofplots import qqplot

        qqplot(residuals, line="s", ax=plt.gca())
        plt.title("Q-Q Plot")
        plt.grid(True)

        # ACF plot
        plt.subplot(2, 2, 4)
        from statsmodels.graphics.tsaplots import plot_acf

        plot_acf(residuals, lags=20, ax=plt.gca())
        plt.title("Residuals ACF")
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_forecast_evaluation(
        forecasts: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """
        Plot forecast evaluation metrics over time.

        Parameters
        ----------
            forecasts (pd.DataFrame): Forecast results from model
            save_path (str): Optional path to save the figure

        """
        # Calculate rolling metrics
        metrics = forecasts[["error", "abs_error"]].rolling(63).mean()
        metrics.columns = ["Rolling MAE", "Rolling MSE"]

        plt.figure(figsize=(14, 6))

        # Plot error metrics
        plt.plot(metrics.index, metrics["Rolling MAE"], label="Rolling MAE (63 days)")
        plt.plot(metrics.index, metrics["Rolling MSE"], label="Rolling MSE (63 days)")

        plt.title("Forecast Error Metrics Over Time")
        plt.xlabel("Date")
        plt.ylabel("Error")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
