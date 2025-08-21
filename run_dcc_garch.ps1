<#
.SYNOPSIS
    Runs the DCC-GARCH model with configurable parameters
.DESCRIPTION
    This script executes the enhanced DCC-GARCH model with options for:
    - Univariate GARCH parameters (p, q)
    - DCC parameters
    - Regularization methods
    - Visualization outputs
#>

param (
    [string]$DataPath = "./data/asset_returns.csv",
    [int]$UnivariateP = 1,
    [int]$UnivariateQ = 1,
    [string]$Dist = "t",
    [int]$DccP = 1,
    [int]$DccQ = 1,
    [string]$Regularization = "eigen",
    [int]$Window = 252,
    [int]$Horizon = 5,
    [string]$OutputDir = ".\output",
    [switch]$Plot,
    [switch]$Verbose
)

# Set up environment
function Initialize-Environment {
    # Activate virtual environment
    if (-not (Test-Path ".venv")) {
        throw "Virtual environment not found. Create one with 'python -m venv .venv'"
    }

    # For PowerShell Core:
    if ($PSVersionTable.PSEdition -eq "Core") {
        .\.venv\bin\Activate.ps1
    }
    # For Windows PowerShell:
    else {
        .\.venv\Scripts\Activate.ps1
    }

    # Create output directory
    if (-not (Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir | Out-Null
    }

    # Set up logging
    $logFile = Join-Path $OutputDir "dcc_garch_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
    Start-Transcript -Path $logFile -Append
}

# Main execution
function Main {
    try {
        Initialize-Environment

        if ($Verbose) {
            Write-Host "Starting DCC-GARCH analysis with parameters:"
            Write-Host "  Data: $DataPath"
            Write-Host "  Univariate GARCH(p,q): ($UnivariateP,$UnivariateQ)"
            Write-Host "  Distribution: $Dist"
            Write-Host "  DCC(p,q): ($DccP,$DccQ)"
            Write-Host "  Regularization: $Regularization"
            Write-Host "  Rolling window: $Window"
            Write-Host "  Forecast horizon: $Horizon"
        }

        # Prepare Python command
        $pythonCmd = @"
import pandas as pd
from src.models.dcc_garch import DCCGARCH
import matplotlib.pyplot as plt
import os

# Load data
returns = pd.read_csv(r'$DataPath', index_col=0, parse_dates=True)

# Initialize model
model = DCCGARCH(
    univariate_garch_params={'p': $UnivariateP, 'q': $UnivariateQ, 'dist': '$Dist'},
    dcc_params={'p': $DccP, 'q': $DccQ},
    regularization='$Regularization'
)

# Fit model
model.fit(returns)

# Forecast
forecast = model.forecast_correlation(horizon=$Horizon)

# Save forecast
forecast_df = pd.DataFrame(forecast, columns=returns.columns, index=returns.columns)
forecast_path = os.path.join(r'$OutputDir', 'forecast_correlation.csv')
forecast_df.to_csv(forecast_path)

# Rolling correlations
rolling_corrs = model.rolling_correlation(returns, window=$Window)
rolling_path = os.path.join(r'$OutputDir', 'rolling_correlations.csv')
rolling_corrs.to_csv(rolling_path)

# Generate plots if requested
if $($Plot.IsPresent):
    fig1 = model.plot_correlation_evolution()
    fig1.savefig(os.path.join(r'$OutputDir', 'correlation_evolution.png'))
    plt.close(fig1)

    fig2 = model.plot_average_correlation()
    fig2.savefig(os.path.join(r'$OutputDir', 'average_correlation.png'))
    plt.close(fig2)

print("Analysis completed successfully")
"@

        # Execute Python
        if ($Verbose) {
            Write-Host "Executing DCC-GARCH model..."
        }

        $output = $pythonCmd | python

        if ($Verbose) {
            Write-Host $output
            Write-Host "Results saved to $OutputDir"

            # Show output files
            Get-ChildItem $OutputDir | Select-Object Name, LastWriteTime | Format-Table -AutoSize
        }

    } catch {
        Write-Host "Error: $_" -ForegroundColor Red
        exit 1
    } finally {
        Stop-Transcript
    }
}

# Entry point
Main
