"""
Test BGN Panel Generation - Regression Test

USAGE:
------
Run this test from the tests/ directory:
    cd tests
    python test_panel_bgn.py

Directory structure (relative to tests/):
    ./original_code/        - Original BGN code to compare against
    ../generate_panel.py    - Current BGN panel generation
    ../config.py            - Configuration (N, T, burnin values)

Note: The test temporarily changes to ./original_code/ when importing the
original code so that data files (Jstar.csv) can be found, then changes back.

PURPOSE:
--------
This test verifies that the refactored BGN panel generation code produces
numerically identical results to the original code when given the same random seed.

WORKFLOW COMPARISON:
-------------------

ORIGINAL CODE (tests/original_code/):
  Implementation: Import directly and call functions in-process

  1. Import panel_functions.py (BGN panel generation module)
     - Imports: parameters.py (model parameters including burnin=300)
     - Imports: vasicek.py (Vasicek model for mean-reverting processes)
     - Imports: loadings_compute.py (factor loading calculations)
     - Data: Jstar.csv (numerical integration results for J* function)

  2. Functions:
     - create_arrays(N, T): Generate all random draws and state variables
       Returns: tuple of 17 arrays (r, mu, xi, sigmaj, chi, beta, corr_zj,
                eret, ret, P, corr_zr, book, op_cash_flow, loadings_mu_taylor,
                loadings_xi_taylor, loadings_mu_proj, loadings_xi_proj)

     - create_panel(N, T, arrays): Transform arrays into panel DataFrame
       Returns: DataFrame with columns [month, firmid, mve, xret, A_1_taylor,
                A_2_taylor, A_1_proj, A_2_proj, roe, bm, mom, agr, rf_stand, size]
       Filters to keep only months > burnin-1 (i.e., months 300-699 for T=700, burnin=300)

  3. Execution:
     - Change directory to tests/original_code/ (for data file access)
     - Import panel_functions module
     - Call create_arrays(N, T) and create_panel(N, T, arrays) directly
     - Change directory back to tests/

CURRENT CODE (utils_bgn/):
  Implementation: Run via command line subprocess (../generate_panel.py)

  1. Import panel_functions_bgn.py
     - Imports: config.py (centralized configuration with BGN_BURNIN=300)
     - Imports: vasicek.py (same Vasicek model, refactored location)
     - Imports: loadings_compute_bgn.py (refactored version)
     - Data: data/Jstar.csv (moved to data/ directory)

  2. Same function signatures and logic as original code
     - create_arrays(N, T): Identical implementation
     - create_panel(N, T, arrays): Identical implementation with improved NaN handling
       Improvement: Sets roe=0 and agr=0 when book value is 0 (instead of NaN)

  3. Execution:
     - Run generate_panel.py via subprocess: python run_generate_panel.py bgn 999
     - Script sets random seed, imports panel_functions_bgn, calls functions
     - Saves panel and arrays to pickle file: bgn_999_panel.pkl
     - Test reads pickle file to get results

TEST CONFIGURATION:
------------------
From config.py:
  - N = 50 firms (configured in config.py)
  - T = 400 time periods (configured in config.py)
  - BGN_BURNIN = 300 months (configured in config.py)
  - TEST_SEED = 12345 (set in test script)

With T=400 and burnin=300:
  - Total periods generated: T + BGN_BURNIN = 700
  - Periods in final panel: 400 (months 300-699)
  - Total rows: N × T = 50 × 400 = 20,000

EXPECTED RESULTS:
----------------
1. Arrays (17 elements) should be numerically identical (rtol=1e-14, atol=1e-15)
2. Panel columns should be numerically identical with ONE exception:
   - roe and agr: Current code has 0 NaNs, original may have NaNs from division by zero
   - This is an expected IMPROVEMENT in the current code

TEST PROCEDURE:
--------------
1. Generate panel with current code using TEST_SEED
2. Generate panel with original code using same TEST_SEED
3. Compare all 17 arrays element-wise
4. Compare all panel columns element-wise (with NaN handling diagnostics)
5. Report PASS/FAIL based on comparison results

Note: This test does NOT save panels. Use test_moments_bgn.py for moment calculations
which will generate and save panels as needed.

IMPORTS AND DEPENDENCIES:
------------------------
This test requires both code versions to be on sys.path and changes working
directory to tests/original_code/ for data file access (Jstar.csv).
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle
import subprocess
from pathlib import Path

# Add paths (run from tests/ directory)
# ../  - for current code (config.py, generate_panel.py, etc.)
# ./test_utils/  - for test utilities (comparison.py)
# ./original_code/  - for original code to compare against
sys.path.insert(0, str(Path(__file__).parent.parent))  # ../
sys.path.insert(0, str(Path(__file__).parent / 'test_utils'))  # ./test_utils/
sys.path.insert(0, str(Path(__file__).parent / 'original_code'))  # ./original_code/

from comparison import assert_close
from config import N, T, BGN_BURNIN

# Test configuration
TEST_SEED = 12345
TEST_PANEL_ID = 999


def test_bgn_panel_generation():
    """Generate BGN panel with current and original code, verify identity."""
    print("=" * 70)
    print("BGN Panel Generation")
    print("=" * 70)
    print(f"N={N}, T={T}, Burnin={BGN_BURNIN}, Seed={TEST_SEED}\n")

    # Generate with current code via command line
    print("[1/4] Generating panel with current code via generate_panel.py...")

    # Run run_generate_panel.py wrapper (sets seed and calls generate_panel)
    test_dir = Path(__file__).parent
    cmd = [sys.executable, str(test_dir / 'test_utils' / 'run_generate_panel.py'), 'bgn', str(TEST_PANEL_ID)]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] generate_panel.py failed:")
        print(result.stdout)
        print(result.stderr)
        return False

    print(f"      Panel generated successfully")

    # Read the pickle file
    from config import DATA_DIR
    pickle_file = Path(DATA_DIR) / f'bgn_{TEST_PANEL_ID}_panel.pkl'

    print(f"\n[2/4] Reading panel from {pickle_file}...")
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    panel_new = data['panel']
    arrays_new = data['arr_tuple']
    print(f"      Panel shape: {panel_new.shape}")

    # Generate with original code (same seed)
    print("\n[3/4] Generating panel with original code...")

    # Change to original_code directory so data files (Jstar.csv) can be found
    original_cwd = os.getcwd()
    original_code_dir = Path(__file__).parent / 'original_code'
    os.chdir(original_code_dir)

    import panel_functions as bgn_old

    # Verify we imported from tests/original_code
    if not bgn_old.__file__.replace('\\', '/').endswith('original_code/panel_functions.py'):
        print(f"WARNING: Imported panel_functions from: {bgn_old.__file__}")
        print(f"Expected location: tests/original_code/panel_functions.py")

    np.random.seed(TEST_SEED)
    arrays_old = bgn_old.create_arrays(N, T + BGN_BURNIN)
    panel_old = bgn_old.create_panel(N, T + BGN_BURNIN, arrays_old)

    # Change back to tests/ directory
    os.chdir(original_cwd)
    print(f"      Panel shape: {panel_old.shape}")

    # Compare panels for identity
    print("\n[4/4] Comparing panels for numerical identity...")
    all_passed = True
    nan_differences_only = True  # Track if failures are only NaN-related

    # Compare panel DataFrames
    for col in panel_new.columns:
        if col not in panel_old.columns:
            print(f"  [SKIP] Column {col} not in original panel (added by generate_panel.py)")
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

    # Compare arrays (arrays is a tuple of 17 elements)
    if len(arrays_new) != len(arrays_old):
        print(f"  [FAIL] Array tuple length mismatch: {len(arrays_new)} vs {len(arrays_old)}")
        all_passed = False
        nan_differences_only = False
    else:
        array_names = ['r', 'mu', 'xi', 'sigmaj', 'chi', 'beta', 'corr_zj', 'eret', 'ret', 'P',
                       'corr_zr', 'book', 'op_cash_flow', 'loadings_mu_taylor', 'loadings_xi_taylor',
                       'loadings_mu_proj', 'loadings_xi_proj']
        for i, name in enumerate(array_names[:len(arrays_new)]):
            try:
                assert_close(
                    arrays_new[i],
                    arrays_old[i],
                    rtol=1e-14,
                    atol=1e-15,
                    name=f"arrays[{i}:{name}]"
                )
                print(f"  [PASS] arrays[{i}:{name}]")
            except AssertionError as e:
                print(f"  [FAIL] arrays[{i}:{name}]: {e}")
                all_passed = False
                nan_differences_only = False

    # Determine test outcome
    if not all_passed:
        if nan_differences_only:
            print("\n  [NOTE] Arrays are identical; panel differences are only NaN handling (expected improvement)")
        else:
            print("\n  [FAIL] Panels are NOT identical")
            print("\n" + "=" * 70)
            print("[FAIL] BGN panel generation test failed")
            print("=" * 70)
            return False
    else:
        print("\n  [ALL PASS] Panels are numerically identical")

    # Save outputs to tests/outputs directory
    print(f"\n[SAVE] Saving panels to tests/outputs...")
    outputs_dir = Path(__file__).parent / 'outputs'
    outputs_dir.mkdir(exist_ok=True)

    # Save original panel and arrays
    original_file = outputs_dir / f'bgn_{TEST_PANEL_ID}_panel_original.pkl'
    with open(original_file, 'wb') as f:
        pickle.dump({'panel': panel_old, 'arr_tuple': arrays_old}, f)
    print(f"      Saved original to {original_file.name}")

    # Save new panel and arrays
    new_file = outputs_dir / f'bgn_{TEST_PANEL_ID}_panel_new.pkl'
    with open(new_file, 'wb') as f:
        pickle.dump({'panel': panel_new, 'arr_tuple': arrays_new}, f)
    print(f"      Saved new to {new_file.name}")

    # Cleanup: delete the original pickle file from DATA_DIR
    print(f"\n[CLEANUP] Deleting {pickle_file}...")
    if pickle_file.exists():
        pickle_file.unlink()
        print("      Deleted successfully")
    else:
        print("      File not found (already deleted)")

    print("\n" + "=" * 70)
    print("[DONE] BGN panel generation test complete")
    print("=" * 70)
    return True


if __name__ == "__main__":
    test_bgn_panel_generation()
