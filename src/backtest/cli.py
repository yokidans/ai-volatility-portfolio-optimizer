import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.backtest.scenarios import StressScenarios

# Import settings directly
from src.config import settings
from src.infra.logging import logger


def load_sample_data() -> pd.DataFrame:
    """Load sample data for testing if real data isn't available"""
    logger.info("Generating sample data for demonstration...")

    # Create sample date range
    dates = pd.date_range(start="2010-01-01", end="2025-01-01", freq="D")
    n_days = len(dates)

    # Generate sample price data for SPY, TSLA, BND
    np.random.seed(settings.SEED)  # Use settings.SEED

    # Base returns with different volatilities
    spy_returns = np.random.normal(0.0005, 0.01, n_days)
    tsla_returns = np.random.normal(0.001, 0.02, n_days)
    bnd_returns = np.random.normal(0.0002, 0.005, n_days)

    # Convert to prices
    spy_prices = 100 * (1 + spy_returns).cumprod()
    tsla_prices = 50 * (1 + tsla_returns).cumprod()
    bnd_prices = 80 * (1 + bnd_returns).cumprod()

    prices = pd.DataFrame(
        {"SPY": spy_prices, "TSLA": tsla_prices, "BND": bnd_prices}, index=dates
    )

    return prices


def main():
    parser = argparse.ArgumentParser(description="Run stress test scenarios")
    parser.add_argument(
        "--crisis",
        type=str,
        required=True,
        help="Comma-separated list of crises (2008,2020,2022)",
    )
    parser.add_argument(
        "--n_simulations",
        type=int,
        default=100,
        help="Number of Monte Carlo simulations",
    )
    parser.add_argument(
        "--output", type=str, default="stress_test_results.csv", help="Output file path"
    )
    parser.add_argument(
        "--use_sample_data",
        action="store_true",
        help="Use sample data instead of real data",
    )

    args = parser.parse_args()

    print(f"Starting stress test with crises: {args.crisis}")
    print(f"Monte Carlo simulations: {args.n_simulations}")
    print(f"Using sample data: {args.use_sample_data}")

    # Initialize stress tester
    stress_tester = StressScenarios(seed=settings.SEED)

    # Load data
    try:
        if args.use_sample_data:
            prices = load_sample_data()
            print(f"Using sample data with shape: {prices.shape}")
        else:
            # Try to import and use real data fetcher
            try:
                from src.data.fetchers import DataFetcher

                fetcher = DataFetcher()
                raw_data, _ = fetcher.fetch_all_data()
                prices = raw_data.pivot(index="date", columns="ticker", values="close")
                print(f"Loaded real data with shape: {prices.shape}")
            except ImportError:
                print("Data fetcher not available, using sample data")
                prices = load_sample_data()

    except Exception as e:
        print(f"Failed to load data: {e}")
        print("Falling back to sample data...")
        prices = load_sample_data()

    # Calculate returns
    returns = prices.pct_change().dropna()
    print(f"Returns data shape: {returns.shape}")

    # Process requested crises
    crisis_map = {"2008": "gfc_2008", "2020": "covid_2020", "2022": "inflation_2022"}

    requested_crises = args.crisis.split(",")
    results = {}

    for crisis in requested_crises:
        crisis = crisis.strip()
        if crisis not in crisis_map:
            print(f"Warning: Unknown crisis: {crisis}. Available: 2008,2020,2022")
            continue

        crisis_key = crisis_map[crisis]
        print(f"Processing {crisis_key} scenario...")

        try:
            # Apply crisis scenario
            stressed_prices = stress_tester.apply_crisis_scenario(prices, crisis_key)

            # Calculate returns from stressed prices
            stressed_returns = stressed_prices.pct_change().dropna()

            # Calculate portfolio metrics (equal-weighted)
            portfolio_returns = stressed_returns.mean(axis=1)
            max_dd = stress_tester._calculate_max_drawdown(portfolio_returns)

            # Store results
            results[crisis_key] = {"max_drawdown": max_dd, "crisis_name": crisis_key}

            print(f"Successfully applied {crisis_key} scenario. Max DD: {max_dd:.2%}")

        except Exception as e:
            print(f"Failed to process {crisis_key}: {e}")
            import traceback

            traceback.print_exc()

    # Run Monte Carlo simulation
    mc_results = {}
    if not returns.empty:
        try:
            print(
                f"Running Monte Carlo stress test with {args.n_simulations} simulations..."
            )
            mc_results = stress_tester.monte_carlo_stress_test(
                returns, args.n_simulations
            )
            print(f"MC Stress Test - 95% VaR: {mc_results.get('var_95', 0):.2%}")
            print(f"MC Stress Test - 95% CVaR: {mc_results.get('cvar_95', 0):.2%}")
        except Exception as e:
            print(f"Monte Carlo simulation failed: {e}")
            mc_results = {"var_95": 0, "cvar_95": 0}

    # Generate output
    output_data = []
    for crisis_name, result in results.items():
        output_data.append(
            {
                "scenario": crisis_name,
                "max_drawdown": result["max_drawdown"],
                "mc_var_95": mc_results.get("var_95", 0),
                "mc_cvar_95": mc_results.get("cvar_95", 0),
                "status": "completed",
            }
        )

    # Add any failed scenarios
    for crisis in requested_crises:
        crisis = crisis.strip()
        if crisis in crisis_map and crisis_map[crisis] not in results:
            output_data.append(
                {
                    "scenario": crisis_map[crisis],
                    "max_drawdown": 0,
                    "mc_var_95": 0,
                    "mc_cvar_95": 0,
                    "status": "failed",
                }
            )

    # Create DataFrame and save
    if output_data:
        df_output = pd.DataFrame(output_data)

        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df_output.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")

        print("\n" + "=" * 60)
        print("STRESS TEST RESULTS")
        print("=" * 60)
        print(
            df_output.to_string(
                index=False,
                formatter={
                    "max_drawdown": "{:.2%}".format,
                    "mc_var_95": "{:.2%}".format,
                    "mc_cvar_95": "{:.2%}".format,
                },
            )
        )

        if "var_95" in mc_results:
            print(f"\nMonte Carlo Results ({args.n_simulations} simulations):")
            print(f"95% VaR: {mc_results['var_95']:.3%}")
            print(f"95% CVaR: {mc_results['cvar_95']:.3%}")

        print("=" * 60)
    else:
        print("No results generated. Check error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
