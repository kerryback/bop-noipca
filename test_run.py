"""Quick test to verify environment and imports before running main workflow."""
import sys
import os

print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("Directory contents:", os.listdir('.'))

# Test imports
try:
    import numpy as np
    print("✓ numpy imported successfully:", np.__version__)
except Exception as e:
    print("✗ numpy import failed:", e)
    sys.exit(1)

try:
    import pandas as pd
    print("✓ pandas imported successfully:", pd.__version__)
except Exception as e:
    print("✗ pandas import failed:", e)
    sys.exit(1)

try:
    import scipy
    print("✓ scipy imported successfully:", scipy.__version__)
except Exception as e:
    print("✗ scipy import failed:", e)
    sys.exit(1)

try:
    import statsmodels
    print("✓ statsmodels imported successfully:", statsmodels.__version__)
except Exception as e:
    print("✗ statsmodels import failed:", e)
    sys.exit(1)

try:
    import numba
    print("✓ numba imported successfully:", numba.__version__)
except Exception as e:
    print("✗ numba import failed:", e)
    sys.exit(1)

try:
    import joblib
    print("✓ joblib imported successfully:", joblib.__version__)
except Exception as e:
    print("✗ joblib import failed:", e)
    sys.exit(1)

# Try importing config
try:
    from config import get_model_config, N, T
    print("✓ config imported successfully")
    print(f"  N={N}, T={T}")

    # Test getting kp14 config
    config = get_model_config('kp14')
    print("✓ kp14 config loaded:", config['model'])
except Exception as e:
    print("✗ config import failed:", e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Try running generate_panel
try:
    print("\nAttempting to import generate_panel...")
    import generate_panel
    print("✓ generate_panel imported successfully")
except Exception as e:
    print("✗ generate_panel import failed:", e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED - Environment is ready!")
print("="*70)
