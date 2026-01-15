"""
Test GS21 Moment Calculations - Regression Test

USAGE:
------
Run this test from the tests/ directory:
    cd tests
    python test_moments_gs21.py

Directory structure (relative to tests/):
    ./original_code/        - Original GS21 code to compare against
    ../generate_panel.py    - Current GS21 panel generation
    ../calculate_moments.py - Current moment calculation
    ../config.py            - Configuration (N, T, burnin values)

Note: The test temporarily changes to ./original_code/ when importing the
original code so that data files (GS21_solfiles/) can be found, then changes back.

PURPOSE:
--------
This test verifies that the refactored GS21 (Gârleanu-Panageas 2021) SDF
moment calculation code produces numerically identical results to the original
code when given the same panel data.

WORKFLOW COMPARISON:
-------------------

ORIGINAL CODE (tests/original_code/):
  Implementation: Import directly and call functions in-process

  1. Import sdf_compute_gs21.py (GS21 SDF moment calculation module)
     - Imports: parameters_gs21.py (model parameters including gamma, psi, rho)
     - Imports: panel_functions_gs21.py (for accessing panel data)

  2. Function: sdf_compute(N, T, arrays)
     - Purpose: Create a closure that computes SDF moments for any given month
     - Inputs:
       * N: Number of firms
       * T: Total time periods (including burnin)
       * arrays: Tuple of arrays from panel generation

     - Returns: sdf_loop(month, panel_id) function
       * month: Month index (0-based, includes burnin period)
       * panel_id: Panel identifier (0 for this test)
       * Returns: (sdf_ret, max_sr, rp, cond_var)
         - sdf_ret: SDF returns (N x 1 array)
         - max_sr: Maximum Sharpe ratio (scalar)
         - rp: Risk premia (typically 2 x 1 array for factors)
         - cond_var: Conditional variance (2 x 2 matrix)

  3. Computation workflow:
     - Extract firm-month data from arrays for the specified month
     - Compute SDF using recursive utility framework
       SDF involves wealth-consumption ratio and value function derivatives
     - Calculate conditional moments using cross-sectional regression
     - Compute maximum Sharpe ratio from risk premia and variance

  4. Execution:
     - Change directory to tests/original_code/ (for data file access)
     - Import sdf_compute_gs21 module
     - Call sdf_compute(N, T, arrays) to create sdf_loop function
     - Call sdf_loop(month, panel_id) for each month

  5. Key difference from BGN/KP14:
     - GS21 uses recursive utility with risk aversion (gamma) and
       intertemporal elasticity of substitution (psi)
     - SDF computation requires evaluating pre-computed solution functions
     - More complex state space due to heterogeneous investment opportunities

CURRENT CODE (utils_gs21/):
  Implementation: Run from the terminal via subprocess (../calculate_moments.py)

  1. Import sdf_compute_gs21.py
     - Imports: config.py (centralized configuration)
     - Imports: panel_functions_gs21.py (refactored version)

  2. Same function signature and logic as original code
     - sdf_compute(N, T, arrays): Identical implementation
     - Returns same closure sdf_loop(month, panel_id)
     - Same return values: (sdf_ret, max_sr, rp, cond_var)

  3. Execution:
     - Run calculate_moments.py via subprocess: python calculate_moments.py gs21_0
     - Script reads panel from pickle file (gs21_0_panel.pkl)
     - Computes moments using sdf_compute_gs21
     - Saves moments to pickle file: gs21_0_moments.pkl
     - Test reads pickle file to get results

TEST CONFIGURATION:
------------------
From config.py:
  - N = 50 firms (configured in config.py)
  - T = 400 time periods (configured in config.py)
  - GS21_BURNIN = 300 months (configured in config.py)
  - TEST_SEED = 12345 (set in test script)

Test months selected:
  - Month 660 = GS21_BURNIN + 360 (middle of post-burnin period)
  - Month 700 = T + GS21_BURNIN (last month, but may be out of bounds)
  - Month 650 = T + GS21_BURNIN - 50 (near end)

Note: Month indices passed to sdf_loop are 0-based, so we use (month - 1)

EXPECTED RESULTS:
----------------
All moment calculations should be numerically identical:
  1. rp (risk premia): 2x1 array, rtol=1e-12, atol=1e-14
  2. cond_var (conditional variance): 2x2 matrix, rtol=1e-12, atol=1e-14
  3. max_sr (maximum Sharpe ratio): scalar, rtol=1e-12, atol=1e-14

Slightly looser tolerances than panel test because moment calculations involve
matrix operations and may accumulate small numerical errors.

TEST PROCEDURE:
--------------
1. Run generate_panel.py via subprocess with TEST_SEED to create panel
2. Run calculate_moments.py via subprocess to compute moments with current code
3. Read panel pickle (gs21_0_panel.pkl) and moments pickle (gs21_0_moments.pkl)
4. Create SDF compute function with original code using arrays from panel
5. For each test month:
   a. Get moments from pickled moments (current code)
   b. Compute moments with original code: sdf_loop_old(month-1, 0)
   c. Compare rp, cond_var, max_sr
6. Report all comparisons
7. Delete both pickle files (cleanup)

Note: This test is self-contained - it generates its own panel and cleans up after itself.

IMPORTS AND DEPENDENCIES:
------------------------
Requires:
  - Command-line tools: generate_panel.py, calculate_moments.py
  - Original code SDF computation (tests/original_code/sdf_compute_gs21.py)
  - Working directory changed to tests/original_code/ for parameter imports
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle
import subprocess
from pathlib import Path
from random import randint

# Add paths (run from tests/ directory)
# ../  - for current code (config.py, generate_panel.py, etc.)
# ./test_utils/  - for test utilities (comparison.py)
# ./original_code/  - for original code to compare against
sys.path.insert(0, str(Path(__file__).parent.parent))  # ../
sys.path.insert(0, str(Path(__file__).parent / 'test_utils'))  # ./test_utils/
sys.path.insert(0, str(Path(__file__).parent / 'original_code'))  # ./original_code/

from comparison import assert_close
from config import N, T, GS21_BURNIN

# Test configuration
TEST_SEED = 12345


def test_gs21_moment_calculations():
    """Test GS21 moment calculations by generating panel, testing moments, then cleanup."""
    print("=" * 70)
    print("GS21 Moment Calculation Test")
    print("=" * 70)
    print(f"N={N}, T={T}, Burnin={GS21_BURNIN}, Seed={TEST_SEED}\n")

    # Test panel identifier
    TEST_PANEL_ID = 0

    # Change to original_code directory so data files can be found
    original_cwd = os.getcwd()
    original_code_dir = Path(__file__).parent / 'original_code'
    os.chdir(original_code_dir)

    # Import original code for SDF computation
    import sdf_compute_gs21 as sdf_old

    # Change back to tests/ directory
    os.chdir(original_cwd)

    # Generate panel with current code via command line
    print("[1/6] Generating panel with current code via generate_panel.py...")
    test_dir = Path(__file__).parent
    cmd = [sys.executable, str(test_dir / 'test_utils' / 'run_generate_panel.py'), 'gs21', str(TEST_PANEL_ID)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] generate_panel.py failed:")
        print(result.stdout)
        print(result.stderr)
        return False

    print(f"      Panel generated successfully")

    # Calculate moments with current code via command line
    print("\n[2/6] Calculating moments with current code via calculate_moments.py...")
    parent_dir = Path(__file__).parent.parent
    cmd = [sys.executable, str(parent_dir / 'calculate_moments.py'), f'gs21_{TEST_PANEL_ID}']
    result = subprocess.run(cmd, cwd=str(parent_dir), capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] calculate_moments.py failed:")
        print(result.stdout)
        print(result.stderr)
        return False

    print(f"      Moments calculated successfully")

    # Read both pickle files
    from config import DATA_DIR
    panel_file = Path(DATA_DIR) / f'gs21_{TEST_PANEL_ID}_panel.pkl'
    moments_file = Path(DATA_DIR) / f'gs21_{TEST_PANEL_ID}_moments.pkl'

    print(f"\n[3/6] Reading panel from {panel_file}...")
    with open(panel_file, 'rb') as f:
        panel_data = pickle.load(f)
    arrays = panel_data['arr_tuple']
    print(f"      Panel shape: {panel_data['panel'].shape}")

    print(f"\n[4/6] Reading moments from {moments_file}...")
    with open(moments_file, 'rb') as f:
        moments_data = pickle.load(f)

    # Extract moments dictionary and metadata
    moments_new = moments_data['moments']
    start_month = moments_data['start_month']
    end_month = moments_data['end_month']
    print(f"      Moments for {len(moments_new)} months loaded (months {start_month} to {end_month})")

    # Create SDF compute function with original code
    print("\n[5/6] Creating SDF compute function with original code...")
    sdf_loop_old = sdf_old.sdf_compute(N, T + GS21_BURNIN, arrays)

    # Test a few months
    # Note: These are absolute month indices including burnin
    # Test first month, last month, and a random middle month
    middle_month = randint(GS21_BURNIN + 361, GS21_BURNIN + T - 2)
    test_months = [GS21_BURNIN + 360, GS21_BURNIN + T - 1, middle_month]

    print(f"\n[6/6] Testing moments for {len(test_months)} months...")

    all_passed = True
    tests_run = 0

    for month in test_months:
        if month >= T + GS21_BURNIN:
            continue

        print(f"\n  Month {month}:")

        # Get moments from current code (pickled)
        # Note: calculate_moments.py stores moments with absolute month numbers as keys
        if month not in moments_new:
            print(f"    [FAIL] Month {month} not in pickled moments!")
            print(f"           Expected months {start_month} to {end_month}")
            all_passed = False
            continue

        moments_dict_new = moments_new[month]
        rp_new = moments_dict_new['rp']
        cond_var_new = moments_dict_new['cond_var']
        max_sr_new = moments_dict_new['max_sr']

        # Compute with original code
        # Note: sdf_loop expects 0-based month index, panel_id=0
        # To compute moments for month M, we call sdf_loop(M-1, 0)
        sdf_ret_old, max_sr_old, rp_old, cond_var_old = sdf_loop_old(month - 1, 0)

        # Compare rp (risk premia) - 2x1 array
        try:
            assert_close(rp_new, rp_old, rtol=1e-12, atol=1e-14, name=f"rp[{month}]")
            print(f"    [PASS] rp")
        except AssertionError as e:
            print(f"    [FAIL] rp: {e}")
            all_passed = False

        # Compare cond_var (conditional variance) - 2x2 matrix
        try:
            assert_close(cond_var_new, cond_var_old, rtol=1e-12, atol=1e-14, name=f"cond_var[{month}]")
            print(f"    [PASS] cond_var")
        except AssertionError as e:
            print(f"    [FAIL] cond_var: {e}")
            all_passed = False

        # Compare max_sr (maximum Sharpe ratio) - scalar
        try:
            assert_close(
                np.array([max_sr_new]),
                np.array([max_sr_old]),
                rtol=1e-12,
                atol=1e-14,
                name=f"max_sr[{month}]"
            )
            print(f"    [PASS] max_sr")
        except AssertionError as e:
            print(f"    [FAIL] max_sr: {e}")
            all_passed = False

        tests_run += 1

    # Summary
    print(f"\n{'-'*70}")
    if tests_run == 0:
        print(f"  [FAIL] No tests were run! Expected to test {len([m for m in test_months if m < T + GS21_BURNIN])} months")
        all_passed = False
    elif all_passed:
        print(f"  [ALL PASS] All {tests_run} month comparisons passed")
    else:
        print(f"  [FAIL] Some comparisons failed (tested {tests_run} months)")

    # Cleanup: delete both pickle files
    print(f"\n[CLEANUP] Deleting {panel_file}...")
    if panel_file.exists():
        panel_file.unlink()
        print("      Deleted successfully")
    else:
        print("      File not found (already deleted)")

    print(f"[CLEANUP] Deleting {moments_file}...")
    if moments_file.exists():
        moments_file.unlink()
        print("      Deleted successfully")
    else:
        print("      File not found (already deleted)")

    # Delete CSV file created by original code
    csv_file = Path(__file__).parent / 'port_gs.csv'
    print(f"[CLEANUP] Deleting {csv_file}...")
    if csv_file.exists():
        csv_file.unlink()
        print("      Deleted successfully")
    else:
        print("      File not found (already deleted)")

    print("\n" + "=" * 70)
    print("[DONE] GS21 moment calculation test complete")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    test_gs21_moment_calculations()
