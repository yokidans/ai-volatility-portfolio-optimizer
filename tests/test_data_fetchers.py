"""Test module for data fetchers."""

from src.data import fetchers


def test_fetchers_smoke():
    """Smoke test for data fetchers."""
    assert hasattr(fetchers, "fetch_yfinance_data")  # noqa: S101
