#!/usr/bin/env python3
"""
Simple test script to verify basic functionality
"""

import numpy as np
import pandas as pd

# Test settings
print("Testing settings...")
try:
    from src.config import settings

    print(f"✓ SEED: {settings.SEED}")
    print(f"✓ TICKERS: {settings.TICKERS}")
except Exception as e:
    print(f"✗ Settings error: {e}")

# Test scenarios
print("\nTesting scenarios...")
try:
    from src.backtest.scenarios import StressScenarios

    tester = StressScenarios(seed=42)
    print("✓ StressScenarios imported successfully")

    # Create test data
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    prices = pd.DataFrame(
        {
            "SPY": np.linspace(100, 110, len(dates)),
            "TSLA": np.linspace(50, 60, len(dates)),
        },
        index=dates,
    )

    # Test a crisis scenario
    stressed = tester.apply_crisis_scenario(prices, "covid_2020")
    print(f"✓ Crisis scenario applied. Shape: {stressed.shape}")

except Exception as e:
    print(f"✗ Scenarios error: {e}")
    import traceback

    traceback.print_exc()

print("\nTest completed!")
