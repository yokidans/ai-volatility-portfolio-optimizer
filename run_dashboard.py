# run_dashboard.py
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from app.dashboard import ElitePortfolioDashboard

if __name__ == "__main__":
    dashboard = ElitePortfolioDashboard()
    dashboard.run()