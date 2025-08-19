import pytest
import numpy as np
import pandas as pd
from src.models.dcc_garch import DCCGARCH
from src.config import settings

@pytest.fixture
def mock_returns():
    np.random.seed(settings.SEED)
    dates = pd.date_range(start="2020-01-01", periods=100)
    returns = pd.DataFrame({
        'SPY': np.random.normal(0.0005, 0.01, 100),
        'TSLA': np.random.normal(0.001, 0.02, 100),
        'BND': np.random.normal(0.0002, 0.005, 100)
    }, index=dates)
    return returns

def test_dcc_garch_fit(mock_returns):
    model = DCCGARCH()
    model.fit(mock_returns)
    
    assert model.univariate_models is not None
    assert len(model.univariate_models) == 3
    assert model.correlations.shape == (100, 3, 3)
    
    # Check correlation matrix properties
    for corr_matrix in model.correlations:
        assert np.allclose(np.diag(corr_matrix), 1.0)  # Diagonal is 1
        assert np.all(corr_matrix >= -1) and np.all(corr_matrix <= 1)  # Valid correlations
        assert np.allclose(corr_matrix, corr_matrix.T)  # Symmetric

def test_forecast_correlation(mock_returns):
    model = DCCGARCH()
    model.fit(mock_returns)
    forecast = model.forecast_correlation()
    
    assert forecast.shape == (3, 3)
    assert np.allclose(np.diag(forecast), np.diag(forecast).T)  # Symmetric
    eigvals = np.linalg.eigvals(forecast)
    assert np.all(eigvals > 0)  # Positive definite

def test_regularization():
    from src.models.dcc_garch import DCCGARCH
    
    # Create a non-positive definite matrix
    R = np.array([
        [1.0, 0.9, 0.9],
        [0.9, 1.0, 0.9],
        [0.9, 0.9, 1.0]
    ])
    
    # Force it to be non-positive definite
    R[2,2] = -1.0
    
    regularized = DCCGARCH._regularize_correlation(R)
    eigvals = np.linalg.eigvals(regularized)
    assert np.all(eigvals > 0)
    assert np.allclose(np.diag(regularized), 1.0)