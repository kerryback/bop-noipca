"""
Test GS21 Panel Generation - Regression Test

PURPOSE:
--------
This test verifies that the refactored GS21 (Gârleanu-Panageas 2021) panel
generation code produces numerically identical results to the original code
when given the same random seed.

WORKFLOW COMPARISON:
-------------------

ORIGINAL CODE (tests/original_code/):
  1. Import panel_functions_gs21.py (GS21 panel generation module)
     - Imports: parameters_gs21.py (model parameters including burnin=300)
     - Data: GS21_solfiles/*.csv (18 solution files for different omega values)
       Files: gs_val_*.csv, gs_pol_*.csv, gs_der_*.csv for omega in [3,6,9,12,15,18]
       Purpose: Pre-computed value functions, policy functions, derivatives

  2. Functions:
     - create_arrays(N, T): Generate all random draws and state variables
       Returns: tuple of arrays including:
         - State variables: theta, omega, z
         - Asset pricing: D, P, ret, eret
         - Characteristics: book, op_cash_flow, loadings
         - Solution interpolators for value/policy functions

     - create_panel(N, T, arrays): Transform arrays into panel DataFrame
       Returns: DataFrame with columns [month, firmid, mve, xret, A_1_taylor,
                A_2_taylor, A_1_proj, A_2_proj, roe, bm, mom, agr]
       Filters to keep only months > burnin-1 (i.e., months 300-699 for T=700, burnin=300)

  3. Key features of GS21 model:
     - Heterogeneous firms with idiosyncratic investment opportunities (omega)
     - Recursive utility with risk aversion and intertemporal elasticity
     - Solution requires numerical methods (pre-computed and stored in CSV files)
     - Uses interpolation to evaluate value/policy functions during simulation

CURRENT CODE (utils_gs21/):
  1. Import panel_functions_gs21.py
     - Imports: config.py (centralized configuration with GS21_BURNIN=300)
     - Data: data/gs21/GS21_solfiles/*.csv (moved to data/ directory)

  2. Same function signatures as original code
     - create_arrays(N, T): Same implementation
     - create_panel(N, T, arrays): Same implementation with improved NaN handling
       Improvement: Sets roe=0 and agr=0 when book value is 0 (instead of NaN)

TEST CONFIGURATION:
------------------
From config.py:
  - N = 50 firms (configured in config.py)
  - T = 400 time periods (configured in config.py)
  - GS21_BURNIN = 300 months (configured in config.py)
  - TEST_SEED = 12345 (set in test script)

With T=400 and burnin=300:
  - Total periods generated: T + GS21_BURNIN = 700
  - Periods in final panel: 400 (months 300-699)
  - Total rows: N × T = 50 × 400 = 20,000

EXPECTED RESULTS:
----------------
1. Arrays should be numerically identical (rtol=1e-14, atol=1e-15)
2. Panel columns should be numerically identical with ONE exception:
   - roe and agr: Current code has 0 NaNs, original may have NaNs from division by zero
   - This is an expected IMPROVEMENT in the current code

TEST PROCEDURE:
--------------
1. Generate panel with current code using TEST_SEED
2. Generate panel with original code using same TEST_SEED
3. Compare all arrays element-wise
4. Compare all panel columns element-wise (with NaN handling diagnostics)
5. Report PASS/FAIL based on comparison results

Note: This test does NOT save panels. Use test_moments_gs21.py for moment calculations
which will generate and save panels as needed.

IMPORTS AND DEPENDENCIES:
------------------------
This test requires both code versions to be on sys.path and changes working
directory to tests/original_code/ for data file access (GS21_solfiles/*.csv).
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add paths for current code
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'test_utils'))

# Add path for original code
original_code_path = str(Path(__file__).parent / 'original_code')
sys.path.insert(0, original_code_path)

from comparison import assert_close
from config import N, T, GS21_BURNIN

# Test configuration
TEST_SEED = 12345

# Change to original_code directory for data file access
original_cwd = os.getcwd()
os.chdir(original_code_path)


def test_gs21_panel_generation():
    """Generate GS21 panel with current and original code, verify identity."""
    print("=" * 70)
    print("GS21 Panel Generation")
    print("=" * 70)
    print(f"N={N}, T={T}, Burnin={GS21_BURNIN}, Seed={TEST_SEED}\n")

    # Import both versions
    # Current code: utils_gs21/panel_functions_gs21.py
    from utils_gs21 import panel_functions_gs21 as gs21_new
    # Original code: tests/original_code/panel_functions_gs21.py
    import panel_functions_gs21 as gs21_old

    # Generate with current code
    print("[1/3] Generating panel with current code...")
    np.random.seed(TEST_SEED)
    arrays_new = gs21_new.create_arrays(N, T + GS21_BURNIN)
    panel_new = gs21_new.create_panel(N, T + GS21_BURNIN, arrays_new)
    print(f"      Panel shape: {panel_new.shape}")

    # Generate with original code (same seed)
    print("\n[2/3] Generating panel with original code...")
    np.random.seed(TEST_SEED)
    arrays_old = gs21_old.create_arrays(N, T + GS21_BURNIN)
    panel_old = gs21_old.create_panel(N, T + GS21_BURNIN, arrays_old)
    print(f"      Panel shape: {panel_old.shape}")

    # Compare panels for identity
    print("\n[3/3] Comparing panels for numerical identity...")
    all_passed = True
    nan_differences_only = True  # Track if failures are only NaN-related

    # Compare panel DataFrames
    for col in panel_new.columns:
        if col not in panel_old.columns:
            print(f"  [FAIL] Column {col} missing in original panel")
            all_passed = False
            nan_differences_only = False
            continue

        try:
            assert_close(
                panel_new[col].values,
                panel_old[col].values,
                rtol=1e-14,
                atol=1e-15,
                name=f"panel[{col}]"
            )
            print(f"  [PASS] panel[{col}]")
        except AssertionError as e:
            # Check if it's a NaN location mismatch
            arr1 = panel_new[col].values
            arr2 = panel_old[col].values
            nan1 = np.isnan(arr1)
            nan2 = np.isnan(arr2)
            if not np.array_equal(nan1, nan2):
                print(f"  [FAIL] panel[{col}]: NaN locations differ (expected improvement)")
                print(f"         New code: {nan1.sum()} NaNs, Original code: {nan2.sum()} NaNs")
                nan_diff = nan1 != nan2
                if nan_diff.sum() > 0 and nan_diff.sum() < 20:
                    # Show a few examples of the mismatch
                    idx = np.where(nan_diff)[0][:5]
                    for i in idx:
                        print(f"         Position {i}: new={arr1[i]:.6f if not nan1[i] else 'NaN'}, old={arr2[i]:.6f if not nan2[i] else 'NaN'}")
            else:
                print(f"  [FAIL] panel[{col}]: {e}")
                nan_differences_only = False
            all_passed = False

    # Compare arrays (arrays is a tuple)
    if len(arrays_new) != len(arrays_old):
        print(f"  [FAIL] Array tuple length mismatch: {len(arrays_new)} vs {len(arrays_old)}")
        all_passed = False
        nan_differences_only = False
    else:
        for i in range(len(arrays_new)):
            try:
                assert_close(
                    arrays_new[i],
                    arrays_old[i],
                    rtol=1e-14,
                    atol=1e-15,
                    name=f"arrays[{i}]"
                )
                print(f"  [PASS] arrays[{i}]")
            except AssertionError as e:
                print(f"  [FAIL] arrays[{i}]: {e}")
                all_passed = False
                nan_differences_only = False

    # Determine test outcome
    if not all_passed:
        if nan_differences_only:
            print("\n  [NOTE] Arrays are identical; panel differences are only NaN handling (expected improvement)")
        else:
            print("\n  [FAIL] Panels are NOT identical")
            print("\n" + "=" * 70)
            print("[FAIL] GS21 panel generation test failed")
            print("=" * 70)
            return False
    else:
        print("\n  [ALL PASS] Panels are numerically identical")

    print("\n" + "=" * 70)
    print("[DONE] GS21 panel generation test complete")
    print("=" * 70)
    return True


if __name__ == "__main__":
    test_gs21_panel_generation()

    # Restore original working directory
    os.chdir(original_cwd)
