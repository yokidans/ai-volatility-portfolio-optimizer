from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import settings
from src.infra.logging import logger


class PortfolioDashboard:
    """Interactive dashboard for portfolio analysis and visualization."""

    def __init__(self):
        self.data_dir = settings.PROCESSED_DATA_DIR
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if "regime_filter" not in st.session_state:
            st.session_state.regime_filter = "All"
        if "risk_appetite" not in st.session_state:
            st.session_state.risk_appetite = "Medium"

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load processed data from disk."""
        try:
            return {
                "returns": pd.read_parquet(self.data_dir / "returns.parquet"),
                "volatility": pd.read_parquet(self.data_dir / "volatility.parquet"),
                "weights": pd.read_parquet(self.data_dir / "portfolio_weights.parquet"),
                "performance": pd.read_parquet(
                    self.data_dir / "backtest_results.parquet"
                ),
            }
        except Exception as e:
            logger.error("Failed to load data", error=str(e))
            st.error("Failed to load data. Please ensure the pipeline has been run.")
            return None

    def render_sidebar(self):
        """Render dashboard sidebar controls."""
        with st.sidebar:
            st.header("Portfolio Controls")
            st.session_state.regime_filter = st.selectbox(
                "Volatility Regime", ["All", "Low", "Medium", "High", "Extreme"]
            )

            st.session_state.risk_appetite = st.select_slider(
                "Risk Appetite", options=["Low", "Medium", "High"], value="Medium"
            )

            st.slider(
                "Lookback Period (months)",
                min_value=1,
                max_value=60,
                value=24,
                key="lookback",
            )

    def render_performance_tab(self, data: Dict[str, pd.DataFrame]):
        """Render portfolio performance visualization."""
        st.header("Portfolio Performance")

        # Filter data based on regime
        if st.session_state.regime_filter != "All":
            regime_data = pd.read_parquet(self.data_dir / "regime_features.parquet")
            regime_mask = (
                regime_data[f"regime_{st.session_state.regime_filter.upper()}"] == 1
            )
            filtered_perf = data["performance"][regime_mask]
        else:
            filtered_perf = data["performance"]

        # Plot cumulative returns
        fig = px.line(
            filtered_perf,
            x=filtered_perf.index,
            y="portfolio_value",
            title="Portfolio Value Over Time",
            labels={"portfolio_value": "Portfolio Value (USD)", "index": "Date"},
        )
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown analysis
        peak = filtered_perf["portfolio_value"].cummax()
        drawdown = (filtered_perf["portfolio_value"] - peak) / peak
        fig = px.area(
            drawdown,
            title="Portfolio Drawdown",
            labels={"value": "Drawdown", "index": "Date"},
        )
        st.plotly_chart(fig, use_container_width=True)

    def render_risk_tab(self, data: Dict[str, pd.DataFrame]):
        """Render risk analysis visualization."""
        st.header("Risk Analysis")

        # CVaR decomposition
        st.subheader("CVaR Contribution by Asset")
        cvar_data = pd.read_parquet(self.data_dir / "cvar_decomposition.parquet")
        fig = px.bar(
            cvar_data,
            x=cvar_data.index,
            y="cvar_contribution",
            title="CVaR Contribution by Asset",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volatility surface
        st.subheader("Volatility Forecasts")
        fig = px.line(
            data["volatility"],
            x=data["volatility"].index,
            y=data["volatility"].columns,
            title="Volatility Forecasts",
            labels={"value": "Volatility", "index": "Date"},
        )
        st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Main dashboard execution method."""
        st.set_page_config(
            page_title="AI Portfolio Optimizer",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("AI-Driven Portfolio Optimization Dashboard")
        self.render_sidebar()

        data = self.load_data()
        if data is None:
            return

        tab1, tab2, tab3 = st.tabs(
            ["Performance", "Risk Analysis", "Portfolio Composition"]
        )

        with tab1:
            self.render_performance_tab(data)

        with tab2:
            self.render_risk_tab(data)

        with tab3:
            st.header("Portfolio Composition Over Time")
            fig = px.area(
                data["weights"],
                x=data["weights"].index,
                y=data["weights"].columns,
                title="Portfolio Weights",
                labels={"value": "Weight", "index": "Date"},
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    dashboard = PortfolioDashboard()
    dashboard.run()
