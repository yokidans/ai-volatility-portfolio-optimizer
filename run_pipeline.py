# run_pipeline.py
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from data.make_dataset import process_raw_data
    process_raw_data()
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying standalone approach...")
    
    # Fallback to standalone processing
    import subprocess
    import os
    
    # Run the standalone script
    standalone_script = Path(__file__).parent / "run_pipeline_standalone.py"
    if standalone_script.exists():
        result = subprocess.run([sys.executable, str(standalone_script)], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    else:
        print("Standalone script not found. Please create run_pipeline_standalone.py")