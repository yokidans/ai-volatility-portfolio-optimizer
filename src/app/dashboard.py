from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import scipy.stats as stats
import warnings
import sys
import logging
from pathlib import Path

# Add the parent directory to Python path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from config.settings import settings
    from infra.logging import logger
except ImportError:
    # Fallback if the proper modules aren't available
    logger = logging.getLogger("dashboard")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    # Simple settings implementation
    class Settings:
        def __init__(self):
            self.BASE_DIR = Path(__file__).parent.parent
            self.DATA_DIR = self.BASE_DIR / "data"
            self.PROCESSED_DIR = self.DATA_DIR / "processed"
    
    settings = Settings()

warnings.filterwarnings('ignore')


class ElitePortfolioDashboard:
    """Advanced interactive dashboard for elite portfolio analysis and visualization."""

    def __init__(self):
        # FIX: Use PROCESSED_DIR instead of PROCESSED_DATA_DIR
        self.data_dir = settings.PROCESSED_DIR
        self.initialize_session_state()
        self.performance_metrics = {}
        self.risk_metrics = {}

    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        defaults = {
            "regime_filter": "All",
            "risk_appetite": "Medium",
            "lookback": 24,
            "benchmark": "SPY",
            "confidence_level": 0.95,
            "stress_scenario": "2020-03",
            "factor_analysis_depth": 3,
            "liquidity_constraint": 0.05,
            "transaction_cost": 0.001,
            "tax_rate": 0.25,
            "view_mode": "Professional"  # Professional, Institutional, or Regulatory
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    # Add this method to your ElitePortfolioDashboard class
    def process_data(self):
        """Process raw data into the format needed for the dashboard."""
        try:
            from data.make_dataset import process_raw_data
            process_raw_data()
            st.success("Data processed successfully!")
            return True
        except Exception as e:
            st.error(f"Failed to process data: {e}")
            try:
                from data.make_dataset import create_sample_data
                create_sample_data()
                st.info("Sample data generated instead.")
                return True
            except Exception as e2:
                st.error(f"Failed to generate sample data: {e2}")
                return False

    # Update the load_data method to offer data processing
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load processed data from disk with enhanced datasets."""
        try:
            # Ensure data directory exists
            if not self.data_dir.exists():
                st.error(f"Data directory not found: {self.data_dir}")
                if st.button("Process Data"):
                    if self.process_data():
                        st.rerun()
                return None
            
            data = {}
            
            # Required files
            required_files = {
                "returns": "returns.parquet",
                "volatility": "volatility.parquet", 
                "weights": "portfolio_weights.parquet",
                "performance": "backtest_results.parquet"
            }
            
            # Check if required files exist
            missing_files = []
            for key, filename in required_files.items():
                file_path = self.data_dir / filename
                if not file_path.exists():
                    missing_files.append(filename)
            
            if missing_files:
                st.error(f"Missing data files: {', '.join(missing_files)}")
                if st.button("Process Data"):
                    if self.process_data():
                        st.rerun()
                return None
            
            # Load required files
            for key, filename in required_files.items():
                file_path = self.data_dir / filename
                data[key] = pd.read_parquet(file_path)
            
            # Try to load optional datasets
            optional_files = {
                "regime_features": "regime_features.parquet",
                "cvar_decomposition": "cvar_decomposition.parquet",
                "factor_exposures": "factor_exposures.parquet",
                "liquidity_metrics": "liquidity_metrics.parquet",
                "correlation_matrix": "correlation_matrix.parquet",
                "stress_test_results": "stress_test_results.parquet",
                "benchmark_returns": "benchmark_returns.parquet",
                "sentiment_data": "sentiment_data.parquet"
            }
            
            for key, filename in optional_files.items():
                file_path = self.data_dir / filename
                if file_path.exists():
                    data[key] = pd.read_parquet(file_path)
                else:
                    logger.warning(f"Optional data file not found: {filename}")
                    data[key] = None
                        
            return data
        except Exception as e:
            logger.error("Failed to load data", error=str(e))
            st.error(f"Failed to load data: {str(e)}")
            if st.button("Try to Process Data"):
                if self.process_data():
                    st.rerun()
            return None

    def render_sidebar(self):
        """Render advanced dashboard sidebar controls."""
        with st.sidebar:
            st.header("🧠 Portfolio Controls")
            
            # View mode selector
            st.session_state.view_mode = st.selectbox(
                "View Mode", 
                ["Professional", "Institutional", "Regulatory"],
                help="Professional: Standard view, Institutional: Advanced metrics, Regulatory: Compliance focus"
            )
            
            # Risk management
            st.subheader("Risk Management")
            st.session_state.regime_filter = st.selectbox(
                "Volatility Regime", ["All", "Low", "Medium", "High", "Extreme"]
            )
            
            st.session_state.risk_appetite = st.select_slider(
                "Risk Appetite", options=["Conservative", "Moderate", "Aggressive", "Custom"], value="Moderate"
            )
            
            if st.session_state.risk_appetite == "Custom":
                st.slider("CVaR Limit (%)", 1.0, 10.0, 5.0, 0.1, key="cvar_limit")
                st.slider("Max Drawdown Limit (%)", 5.0, 50.0, 20.0, 1.0, key="max_drawdown_limit")
            
            st.session_state.confidence_level = st.slider(
                "Confidence Level", 0.85, 0.99, 0.95, 0.01,
                help="Confidence level for VaR/CVaR calculations"
            )
            
            # Portfolio constraints
            st.subheader("Portfolio Constraints")
            st.session_state.liquidity_constraint = st.slider(
                "Liquidity Constraint (%)", 0.01, 0.20, 0.05, 0.01,
                help="Minimum liquidity threshold for position sizing"
            )
            
            st.session_state.transaction_cost = st.slider(
                "Transaction Cost (%)", 0.0, 0.02, 0.001, 0.0005,
                help="Estimated transaction costs as percentage of trade value"
            )
            
            st.session_state.tax_rate = st.slider(
                "Tax Rate (%)", 0.0, 0.50, 0.25, 0.01,
                help="Estimated tax impact on realized gains"
            )
            
            # Analysis parameters
            st.subheader("Analysis Parameters")
            st.session_state.lookback = st.slider(
                "Lookback Period (months)", 1, 120, 24, 1
            )
            
            stress_scenarios = ["2008-09", "2020-03", "2015-08", "2018-12", "Custom"]
            st.session_state.stress_scenario = st.selectbox(
                "Stress Test Scenario", stress_scenarios
            )
            
            if st.session_state.stress_scenario == "Custom":
                st.date_input("Custom Stress Start Date", value=datetime(2020, 3, 1))
                st.date_input("Custom Stress End Date", value=datetime(2020, 4, 1))
            
            # Benchmark selection
            benchmarks = ["SPY", "QQQ", "IWM", "AGG", "Custom"]
            st.session_state.benchmark = st.selectbox("Benchmark", benchmarks)
            
            if st.session_state.benchmark == "Custom":
                st.text_input("Custom Benchmark Ticker", value="SPY")
            
            # Advanced settings expander
            with st.expander("Advanced Settings"):
                st.checkbox("Enable Real-time Data", value=False)
                st.checkbox("Show Monte Carlo Simulations", value=True)
                st.checkbox("Include ESG Factors", value=True)
                st.checkbox("Show Attribution Analysis", value=True)
                st.slider("Factor Analysis Depth", 1, 10, 3, 1, key="factor_analysis_depth")

    def calculate_advanced_metrics(self, data: Dict[str, pd.DataFrame]):
        """Calculate advanced performance and risk metrics."""
        returns = data["returns"]
        perf = data["performance"]
        
        # Calculate basic metrics
        total_return = (perf["portfolio_value"].iloc[-1] / perf["portfolio_value"].iloc[0] - 1) * 100
        annualized_return = (1 + total_return/100) ** (252/len(perf)) - 1
        
        # Calculate volatility
        daily_returns = perf["portfolio_value"].pct_change().dropna()
        annualized_vol = daily_returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate for simplicity)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
        
        # Calculate Sortino ratio (only downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_dev if downside_dev != 0 else 0
        
        # Calculate maximum drawdown
        peak = perf["portfolio_value"].cummax()
        drawdown = (perf["portfolio_value"] - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Calculate VaR and CVaR
        var = np.percentile(daily_returns, (1 - st.session_state.confidence_level) * 100) * 100
        cvar = daily_returns[daily_returns <= var/100].mean() * 100
        
        # Calculate tail risk metrics
        skewness = stats.skew(daily_returns)
        kurtosis = stats.kurtosis(daily_returns)
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown/100) if max_drawdown != 0 else 0
        
        # Calculate information ratio if benchmark is available
        if "benchmark_returns" in data and data["benchmark_returns"] is not None:
            benchmark_returns = data["benchmark_returns"]
            excess_returns = daily_returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error != 0 else 0
        else:
            information_ratio = None
        
        # Store metrics
        self.performance_metrics = {
            "Total Return": f"{total_return:.2f}%",
            "Annualized Return": f"{annualized_return*100:.2f}%",
            "Annualized Volatility": f"{annualized_vol*100:.2f}%",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Sortino Ratio": f"{sortino_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2f}%",
            f"VaR ({st.session_state.confidence_level*100:.0f}%)": f"{var:.2f}%",
            f"CVaR ({st.session_state.confidence_level*100:.0f}%)": f"{cvar:.2f}%",
            "Skewness": f"{skewness:.2f}",
            "Kurtosis": f"{kurtosis:.2f}",
            "Calmar Ratio": f"{calmar_ratio:.2f}",
        }
        
        if information_ratio is not None:
            self.performance_metrics["Information Ratio"] = f"{information_ratio:.2f}"
            
        # Calculate risk metrics
        self.risk_metrics = self.calculate_risk_decomposition(data)

    def calculate_risk_decomposition(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate risk decomposition metrics."""
        # This would typically involve factor risk model calculations
        # For simplicity, we'll use placeholder values
        return {
            "Market Risk": 65.2,
            "Interest Rate Risk": 12.4,
            "Currency Risk": 8.7,
            "Credit Risk": 7.3,
            "Liquidity Risk": 4.1,
            "Other": 2.3
        }

    def render_performance_tab(self, data: Dict[str, pd.DataFrame]):
        """Render advanced portfolio performance visualization."""
        st.header("📈 Portfolio Performance")
        
        # Calculate and display performance metrics
        self.calculate_advanced_metrics(data)
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", self.performance_metrics["Total Return"])
            st.metric("Annualized Return", self.performance_metrics["Annualized Return"])
            
        with col2:
            st.metric("Annualized Volatility", self.performance_metrics["Annualized Volatility"])
            st.metric("Sharpe Ratio", self.performance_metrics["Sharpe Ratio"])
            
        with col3:
            st.metric("Max Drawdown", self.performance_metrics["Max Drawdown"])
            st.metric("Sortino Ratio", self.performance_metrics["Sortino Ratio"])
            
        with col4:
            st.metric(f"VaR ({st.session_state.confidence_level*100:.0f}%)", 
                     self.performance_metrics[f"VaR ({st.session_state.confidence_level*100:.0f}%)"])
            st.metric(f"CVaR ({st.session_state.confidence_level*100:.0f}%)", 
                     self.performance_metrics[f"CVaR ({st.session_state.confidence_level*100:.0f}%)"])

        # Filter data based on regime
        if st.session_state.regime_filter != "All" and "regime_features" in data and data["regime_features"] is not None:
            regime_mask = (
                data["regime_features"][f"regime_{st.session_state.regime_filter.upper()}"] == 1
            )
            filtered_perf = data["performance"][regime_mask]
        else:
            filtered_perf = data["performance"]

        # Create subplots for performance visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Portfolio Value", "Drawdown", "Rolling Sharpe (1Y)", "Return Distribution"),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Portfolio value
        fig.add_trace(
            go.Scatter(x=filtered_perf.index, y=filtered_perf["portfolio_value"], 
                      name="Portfolio Value", line=dict(color='#636EFA')),
            row=1, col=1
        )
        
        # Drawdown
        peak = filtered_perf["portfolio_value"].cummax()
        drawdown = (filtered_perf["portfolio_value"] - peak) / peak
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown, name="Drawdown", 
                      fill='tozeroy', line=dict(color='#EF553B')),
            row=1, col=2
        )
        
        # Rolling Sharpe ratio (1 year)
        daily_returns = filtered_perf["portfolio_value"].pct_change().dropna()
        rolling_sharpe = daily_returns.rolling(252).mean() / daily_returns.rolling(252).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name="Rolling Sharpe", 
                      line=dict(color='#00CC96')),
            row=2, col=1
        )
        
        # Return distribution
        hist_data = np.random.normal(daily_returns.mean(), daily_returns.std(), 10000)
        fig.add_trace(
            go.Histogram(x=daily_returns, name="Returns", nbinsx=50, 
                        histnorm='probability density', opacity=0.75),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=np.sort(hist_data), 
                      y=stats.norm.pdf(np.sort(hist_data), daily_returns.mean(), daily_returns.std()),
                      name="Normal Dist", line=dict(color='black', dash='dash')),
            row=2, col=2
        )

        fig.update_layout(height=600, showlegend=True, title_text="Portfolio Performance Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance attribution
        st.subheader("Performance Attribution")
        if "factor_exposures" in data and data["factor_exposures"] is not None:
            self.render_performance_attribution(data["factor_exposures"])
        else:
            st.info("Factor exposure data not available for performance attribution")

    def render_performance_attribution(self, factor_exposures: pd.DataFrame):
        """Render performance attribution analysis."""
        # Simplified performance attribution - in practice, this would use a factor model
        col1, col2 = st.columns(2)
        
        with col1:
            # Factor contribution chart
            fig = px.bar(
                factor_exposures.mean().sort_values(ascending=False).head(10),
                title="Top Factor Exposures",
                labels={"value": "Exposure", "index": "Factor"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Sector exposure (placeholder)
            sectors = {
                "Technology": 32.5,
                "Healthcare": 18.7,
                "Financials": 15.2,
                "Consumer Cyclical": 12.4,
                "Industrials": 8.9,
                "Energy": 5.3,
                "Other": 7.0
            }
            fig = px.pie(
                values=list(sectors.values()), 
                names=list(sectors.keys()),
                title="Sector Exposure"
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_risk_tab(self, data: Dict[str, pd.DataFrame]):
        """Render advanced risk analysis visualization."""
        st.header("⚠️ Risk Analysis")
        
        # Display risk metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Risk Decomposition")
            for risk_type, percentage in self.risk_metrics.items():
                st.metric(risk_type, f"{percentage}%")
                
        with col2:
            st.subheader("Concentration Risk")
            # Calculate concentration metrics
            weights = data["weights"].iloc[-1]  # Latest weights
            herfindahl = (weights ** 2).sum() * 100
            top_5_concentration = weights.nlargest(5).sum() * 100
            
            st.metric("Herfindahl Index", f"{herfindahl:.2f}%")
            st.metric("Top 5 Holdings Concentration", f"{top_5_concentration:.2f}%")
            
        with col3:
            st.subheader("Liquidity Risk")
            if "liquidity_metrics" in data and data["liquidity_metrics"] is not None:
                liquidity = data["liquidity_metrics"].iloc[-1]
                st.metric("Avg. Daily Volume (30D)", f"${liquidity.get('avg_daily_volume', 0):,.0f}")
                st.metric("Bid-Ask Spread", f"{liquidity.get('bid_ask_spread', 0)*100:.2f}%")
            else:
                st.info("Liquidity data not available")

        # Create risk visualization subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("CVaR Decomposition", "Volatility Forecast", 
                           "Correlation Heatmap", "Stress Test Results"),
            specs=[[{}, {}], [{}, {}]]
        )

        # CVaR decomposition
        if "cvar_decomposition" in data and data["cvar_decomposition"] is not None:
            cvar_data = data["cvar_decomposition"]
            fig.add_trace(
                go.Bar(x=cvar_data.index, y=cvar_data["cvar_contribution"], 
                      name="CVaR Contribution"),
                row=1, col=1
            )
        else:
            fig.add_annotation(
                x=0.5, y=0.5, xref="x domain", yref="y domain",
                text="CVaR data not available", showarrow=False,
                row=1, col=1
            )

        # Volatility surface
        volatility_data = data["volatility"].iloc[-min(100, len(data["volatility"])):]  # Last 100 days
        for column in volatility_data.columns:
            fig.add_trace(
                go.Scatter(x=volatility_data.index, y=volatility_data[column], 
                          name=column, mode='lines'),
                row=1, col=2
            )

        # Correlation heatmap
        if "correlation_matrix" in data and data["correlation_matrix"] is not None:
            corr_matrix = data["correlation_matrix"]
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
                          colorscale='RdBu_r', zmid=0),
                row=2, col=1
            )
        else:
            fig.add_annotation(
                x=0.5, y=0.5, xref="x domain", yref="y domain",
                text="Correlation data not available", showarrow=False,
                row=2, col=1
            )

        # Stress test results
        if "stress_test_results" in data and data["stress_test_results"] is not None:
            stress_data = data["stress_test_results"]
            fig.add_trace(
                go.Bar(x=stress_data.index, y=stress_data["drawdown"]*100, 
                      name="Stress Test Drawdown"),
                row=2, col=2
            )
        else:
            fig.add_annotation(
                x=0.5, y=0.5, xref="x domain", yref="y domain",
                text="Stress test data not available", showarrow=False,
                row=2, col=2
            )

        fig.update_layout(height=700, title_text="Risk Analysis Dashboard")
        st.plotly_chart(fig, use_container_width=True)
        
        # Advanced risk metrics expander
        with st.expander("Advanced Risk Metrics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Extreme Risk Measures**")
                # Calculate extreme risk metrics (simplified)
                returns = data["performance"]["portfolio_value"].pct_change().dropna()
                st.metric("5% Worst Day Avg", f"{returns.quantile(0.05)*100:.2f}%")
                st.metric("Expected Shortfall (97.5%)", f"{returns[returns <= returns.quantile(0.025)].mean()*100:.2f}%")
                
            with col2:
                st.write("**Liquidation Analysis**")
                # Estimate liquidation time based on volume
                if "liquidity_metrics" in data and data["liquidity_metrics"] is not None:
                    portfolio_value = data["performance"]["portfolio_value"].iloc[-1]
                    avg_daily_volume = data["liquidity_metrics"].iloc[-1].get("avg_daily_volume", portfolio_value * 0.1)
                    liquidation_days = portfolio_value / (avg_daily_volume * 0.1)  # Assuming 10% of daily volume
                    st.metric("Estimated Liquidation Days", f"{liquidation_days:.1f}")
                else:
                    st.info("Liquidation analysis requires liquidity data")

    def render_composition_tab(self, data: Dict[str, pd.DataFrame]):
        """Render advanced portfolio composition visualization."""
        st.header("🧩 Portfolio Composition")
        
        # Current allocation
        current_weights = data["weights"].iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Current allocation pie chart
            fig = px.pie(
                values=current_weights.values, 
                names=current_weights.index,
                title="Current Allocation"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Turnover analysis
            turnover = data["weights"].diff().abs().sum(axis=1) / 2
            fig = px.line(
                x=turnover.index, y=turnover,
                title="Portfolio Turnover",
                labels={"x": "Date", "y": "Turnover"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col3:
            # Concentration over time
            concentration = (data["weights"] ** 2).sum(axis=1)
            fig = px.line(
                x=concentration.index, y=concentration,
                title="Concentration (Herfindahl)",
                labels={"x": "Date", "y": "Concentration"}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Weight evolution
        st.subheader("Weight Evolution")
        fig = px.area(
            data["weights"],
            x=data["weights"].index,
            y=data["weights"].columns,
            title="Portfolio Weights Over Time",
            labels={"value": "Weight", "index": "Date"},
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimization constraints analysis
        st.subheader("Optimization Constraints Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Constraint adherence
            constraints = {
                "Max Weight Constraint": 95,
                "Min Weight Constraint": 85,
                "Liquidity Constraint": 92,
                "Sector Limit Constraint": 88
            }
            fig = px.bar(
                x=list(constraints.keys()), y=list(constraints.values()),
                title="Constraint Adherence (%)",
                labels={"x": "Constraint", "y": "Adherence (%)"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Transaction cost impact
            costs = {
                "Rebalancing": 0.12,
                "Market Impact": 0.08,
                "Bid-Ask Spread": 0.05,
                "Tax Impact": 0.15
            }
            fig = px.pie(
                values=list(costs.values()), 
                names=list(costs.keys()),
                title="Transaction Cost Breakdown"
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_factor_analysis_tab(self, data: Dict[str, pd.DataFrame]):
        """Render advanced factor analysis."""
        st.header("🔍 Factor Analysis")
        
        if "factor_exposures" not in data or data["factor_exposures"] is None:
            st.warning("Factor exposure data not available")
            return
            
        factor_data = data["factor_exposures"]
        
        # Factor exposure over time
        st.subheader("Factor Exposure Over Time")
        fig = px.line(
            factor_data,
            x=factor_data.index,
            y=factor_data.columns,
            title="Factor Exposures",
            labels={"value": "Exposure", "index": "Date"},
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Factor correlation
        st.subheader("Factor Correlation")
        factor_corr = factor_data.corr()
        fig = px.imshow(
            factor_corr,
            title="Factor Correlation Matrix",
            aspect="auto",
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Factor performance attribution
        st.subheader("Factor Performance Attribution")
        col1, col2 = st.columns(2)
        
        with col1:
            # Factor contribution to returns
            factor_contrib = factor_data.mean() * factor_data.std()
            fig = px.bar(
                x=factor_contrib.index, y=factor_contrib.values,
                title="Factor Contribution to Returns",
                labels={"x": "Factor", "y": "Contribution"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Factor risk contribution
            factor_risk = (factor_data ** 2).mean()
            fig = px.pie(
                values=factor_risk.values, 
                names=factor_risk.index,
                title="Factor Risk Contribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_scenario_analysis_tab(self, data: Dict[str, pd.DataFrame]):
        """Render scenario analysis and stress testing."""
        st.header("🌪️ Scenario Analysis")
        
        # Stress test results
        if "stress_test_results" in data and data["stress_test_results"] is not None:
            stress_data = data["stress_test_results"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    x=stress_data.index, y=stress_data["drawdown"]*100,
                    title="Stress Test Results - Maximum Drawdown",
                    labels={"x": "Scenario", "y": "Drawdown (%)"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                fig = px.bar(
                    x=stress_data.index, y=stress_data["recovery_days"],
                    title="Stress Test Results - Recovery Days",
                    labels={"x": "Scenario", "y": "Days to Recover"}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Stress test data not available")
            
        # Monte Carlo simulation (placeholder)
        st.subheader("Monte Carlo Simulation")
        
        # Generate sample Monte Carlo paths
        np.random.seed(42)
        n_paths = 100
        n_days = 252
        returns = data["performance"]["portfolio_value"].pct_change().dropna()
        
        mu = returns.mean()
        sigma = returns.std()
        
        # Generate random paths
        paths = np.zeros((n_days, n_paths))
        paths[0] = data["performance"]["portfolio_value"].iloc[-1]  # Start from current value
        
        for i in range(1, n_days):
            shock = np.random.normal(mu, sigma, n_paths)
            paths[i] = paths[i-1] * (1 + shock)
            
        # Plot Monte Carlo simulation
        fig = go.Figure()
        for i in range(n_paths):
            fig.add_trace(go.Scatter(
                x=list(range(n_days)), 
                y=paths[:, i],
                mode='lines',
                line=dict(width=0.5, color='gray'),
                showlegend=False
            ))
            
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=list(range(n_days)),
            y=np.percentile(paths, 95, axis=1),
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(n_days)),
            y=np.percentile(paths, 5, axis=1),
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            showlegend=False
        ))
        
        fig.update_layout(
            title="Monte Carlo Simulation - 100 Portfolio Paths (1 Year)",
            xaxis_title="Days",
            yaxis_title="Portfolio Value"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Probability of loss
        final_values = paths[-1]
        prob_loss = np.mean(final_values < paths[0])
        prob_20_percent_loss = np.mean(final_values < paths[0] * 0.8)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probability of Loss", f"{prob_loss*100:.1f}%")
        with col2:
            st.metric("Probability of >20% Loss", f"{prob_20_percent_loss*100:.1f}%")

    def run(self):
        """Main dashboard execution method."""
        st.set_page_config(
            page_title="AI Portfolio Optimizer - Elite Edition",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("🎯 AI-Driven Portfolio Optimization Dashboard ")
        self.render_sidebar()

        data = self.load_data()
        if data is None:
            return

        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Performance", "Risk Analysis", "Portfolio Composition", "Factor Analysis", "Scenario Analysis"]
        )

        with tab1:
            self.render_performance_tab(data)

        with tab2:
            self.render_risk_tab(data)

        with tab3:
            self.render_composition_tab(data)
            
        with tab4:
            self.render_factor_analysis_tab(data)
            
        with tab5:
            self.render_scenario_analysis_tab(data)


if __name__ == "__main__":
    dashboard = ElitePortfolioDashboard()
    dashboard.run()