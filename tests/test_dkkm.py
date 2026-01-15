"""
Test DKKM Factor Computation - Regression Test

USAGE:
------
Run this test from the tests/ directory:
    cd tests
    python test_dkkm.py <model> <nfeatures>

where:
    <model> is one of: bgn, kp14, gs21
    <nfeatures> is the number of random features (e.g., 6, 36, 360)

Examples:
    python test_dkkm.py bgn 360
    python test_dkkm.py kp14 180
    python test_dkkm.py gs21 6

Directory structure (relative to tests/):
    ./original_code/        - Original DKKM code to compare against
    ../run_dkkm.py          - Current DKKM factor computation
    ../config.py            - Configuration (N, T, burnin values)

PURPOSE:
--------
This test verifies that the refactored DKKM (Deep Kernel Kernel Methods) factor
computation produces numerically identical results to the original code when
given the same panel data and random weight matrix.

WORKFLOW COMPARISON:
-------------------

ORIGINAL CODE (tests/original_code/):
  Implementation: Import directly and call functions in-process

  1. Import dkkm_functions.py (original DKKM module)
     - Imports: parameters.py (model parameters)
     - Uses joblib.Parallel for parallelization

  2. Functions:
     - rff(data, rf, W, model): Random Fourier Features for single month
       Returns: rank-standardized weights, non-rank-standardized weights

     - factors(panel, W, n_jobs, start, end, model, chars): Panel of RFF returns
       Returns: (f_rs, f_nors) - DataFrames of factor returns

  3. Execution:
     - Import dkkm_functions module
     - Call factors() with panel data
     - Returns factor DataFrames directly

CURRENT CODE (utils_factors/):
  Implementation: Run from the terminal via subprocess (../run_dkkm.py)

  1. Import dkkm_functions.py (refactored version)
     - Imports: config.py (centralized configuration)
     - Uses numba acceleration (dkkm_functions_numba.py)
     - Improved memory efficiency

  2. Same function signatures as original code
     - Identical mathematical implementation
     - Performance optimizations with numba
     - Vectorized operations where possible

  3. Execution:
     - Run run_dkkm.py via subprocess: python run_dkkm.py bgn_0 360
     - Script reads panel from pickle file (bgn_0_panel.pkl)
     - Computes DKKM factors using dkkm_functions
     - Saves factors to pickle file: bgn_0_dkkm_360.pkl
     - Test reads pickle file to get results

TEST CONFIGURATION:
------------------
From config.py:
  - N = 50 firms
  - T = 400 time periods
  - BGN_BURNIN/KP14_BURNIN/GS21_BURNIN = 300 months
  - TEST_SEED = 12345 (set in test script)
  - Random weight matrix W is generated with fixed seed

EXPECTED RESULTS:
----------------
All DKKM factor calculations should be numerically identical:
  1. f_rs (rank-standardized): DataFrame, rtol=1e-14, atol=1e-15
  2. f_nors (non-rank-standardized): DataFrame, rtol=1e-14, atol=1e-15

TEST PROCEDURE:
--------------
1. Generate panel with current code via run_generate_panel.py subprocess
2. Read panel pickle file
3. Generate random weight matrix W with fixed seed
4. Compute DKKM factors with original code using panel data and W
5. Compute DKKM factors with current code via run_dkkm.py subprocess
6. Read DKKM pickle file from current code
7. Compare original vs current factor returns
8. Delete all pickle files (cleanup)
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle
import subprocess
import random
from pathlib import Path

# Add paths (run from tests/ directory)
# ../  - for current code (config.py, run_dkkm.py, etc.)
# ./test_utils/  - for test utilities (comparison.py)
# ./original_code/  - for original code to compare against
sys.path.insert(0, str(Path(__file__).parent.parent))  # ../
sys.path.insert(0, str(Path(__file__).parent / 'test_utils'))  # ./test_utils/
sys.path.insert(0, str(Path(__file__).parent / 'original_code'))  # ./original_code/

from comparison import assert_close
from config import N, T

# Test configuration
TEST_SEED = 12345
EXPECTED_N = 50
EXPECTED_T = 400


def check_config_values():
    """Check if N and T match expected test values."""
    if N != EXPECTED_N or T != EXPECTED_T:
        print("\n" + "!" * 70)
        print("WARNING: Configuration mismatch detected!")
        print("!" * 70)
        print(f"Expected: N={EXPECTED_N}, T={EXPECTED_T}")
        print(f"Current:  N={N}, T={T}")
        print("\nTests are designed for N=50 and T=400.")
        print("Running with different values may cause tests to fail or produce")
        print("unexpected results.")
        print()

        response = input("Do you want to continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("\nTest aborted by user.")
            return False
        print()
    return True


def prepare_panel_for_dkkm(panel, chars):
    """Prepare panel data for DKKM factor computation."""
    # Set multi-index
    panel = panel.set_index(['month', 'firmid'])

    # Get valid data range
    start = panel.index.get_level_values('month').min()
    end = panel.index.get_level_values('month').max()

    return panel, start, end


def compute_dkkm_with_original(panel, W, nfeatures, chars, start, end, model, n_jobs=1):
    """Compute DKKM factors using original code."""
    import dkkm_functions as dkkm_old

    print("Computing DKKM factors with original code...")

    # Compute DKKM factors
    f_rs_old, f_nors_old = dkkm_old.factors(
        panel=panel,
        W=W,
        n_jobs=n_jobs,
        start=start,
        end=end,
        model=model,
        chars=chars
    )

    print(f"  Original f_rs shape: {f_rs_old.shape}")
    print(f"  Original f_nors shape: {f_nors_old.shape}")

    return f_rs_old, f_nors_old


def test_dkkm_factors(model, nfeatures):
    """Test DKKM factor computation for a given model and number of features."""
    # Get burnin for this model
    from config import BGN_BURNIN, KP14_BURNIN, GS21_BURNIN
    burnin_map = {'bgn': BGN_BURNIN, 'kp14': KP14_BURNIN, 'gs21': GS21_BURNIN}
    burnin = burnin_map[model]

    print("=" * 70)
    print(f"DKKM Factor Test: {model.upper()}")
    print("=" * 70)
    print(f"N={N}, T={T}, Burnin={burnin}, Seed={TEST_SEED}")
    print(f"Number of features: {nfeatures}\n")

    # Test panel identifier
    TEST_PANEL_ID = 0
    panel_id = f"{model}_{TEST_PANEL_ID}"

    # Get model configuration
    chars = ['size', 'bm', 'agr', 'roe', 'mom']

    # Step 1: Generate panel with current code via command line
    print("[1/5] Generating panel with current code via generate_panel.py...")
    test_dir = Path(__file__).parent
    cmd = [sys.executable, str(test_dir / 'test_utils' / 'run_generate_panel.py'), model, str(TEST_PANEL_ID)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] generate_panel.py failed:")
        print(result.stdout)
        print(result.stderr)
        return False

    print(f"      Panel generated successfully")

    # Step 2: Read panel pickle file
    from config import DATA_DIR
    panel_file = Path(DATA_DIR) / f'{panel_id}_panel.pkl'

    print(f"\n[2/7] Reading panel from {panel_file}...")
    with open(panel_file, 'rb') as f:
        panel_data = pickle.load(f)
    panel = panel_data['panel']
    print(f"      Panel shape: {panel.shape}")

    # Get parent directory for running scripts
    parent_dir = Path(__file__).parent.parent

    # Step 3: Generate moments file (required for portfolio statistics)
    print(f"\n[3/7] Generating moments file via calculate_moments.py...")
    cmd = [sys.executable, str(parent_dir / 'calculate_moments.py'), panel_id]
    result = subprocess.run(cmd, cwd=str(parent_dir), capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] calculate_moments.py failed:")
        print(result.stdout)
        print(result.stderr)
        return False

    print(f"      Moments file generated successfully")
    moments_file = Path(DATA_DIR) / f'{panel_id}_moments.pkl'

    # Step 4: Compute DKKM factors with current code via run_dkkm.py
    print(f"\n[4/7] Computing DKKM factors with current code via run_dkkm.py...")
    parent_dir = Path(__file__).parent.parent
    cmd = [sys.executable, str(parent_dir / 'run_dkkm.py'), panel_id, str(nfeatures)]
    result = subprocess.run(cmd, cwd=str(parent_dir), capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] run_dkkm.py failed:")
        print(result.stdout)
        print(result.stderr)
        return False

    print(f"      DKKM factors computed successfully")

    # Read DKKM pickle file and extract W matrix
    dkkm_file = Path(DATA_DIR) / f'{panel_id}_dkkm_{nfeatures}.pkl'
    print(f"\n[5/7] Reading DKKM factors from {dkkm_file}...")
    with open(dkkm_file, 'rb') as f:
        dkkm_data = pickle.load(f)

    # Extract factors - current code saves only one version based on config
    if 'dkkm_factors' not in dkkm_data:
        print(f"[ERROR] Could not find dkkm_factors in DKKM output")
        print(f"      Available keys: {list(dkkm_data.keys())}")
        return False

    factors_new = dkkm_data['dkkm_factors']
    rank_standardize = dkkm_data.get('rank_standardize', True)  # Default to True

    # Extract W matrix used by run_dkkm.py
    if 'weights' not in dkkm_data:
        print(f"[ERROR] Could not find W matrix ('weights') in DKKM output")
        print(f"      Available keys: {list(dkkm_data.keys())}")
        return False

    W = dkkm_data['weights']

    print(f"      Current factors shape: {factors_new.shape}")
    print(f"      Rank standardize: {rank_standardize}")
    print(f"      W matrix shape: {W.shape}")

    # Step 6: Compute DKKM factors with original code using the same W
    print(f"\n[6/7] Computing DKKM factors with original code using same W matrix...")
    panel_prepared, start, end = prepare_panel_for_dkkm(panel.copy(), chars)
    f_rs_old, f_nors_old = compute_dkkm_with_original(
        panel_prepared, W, nfeatures, chars, start, end, model
    )

    # Step 7: Verify portfolio statistics against manual computation
    print(f"\n[7/7] Verifying portfolio statistics using original formulas...")
    stats_passed = True

    if 'dkkm_stats' not in dkkm_data:
        print(f"      [FAIL] 'dkkm_stats' not found in pickle file")
        stats_passed = False
    else:
        dkkm_stats = dkkm_data['dkkm_stats']
        print(f"      DKKM stats shape: {dkkm_stats.shape}")
        print(f"      DKKM stats columns: {list(dkkm_stats.columns)}")

        # Load SDF moments to manually compute stats
        moments_file = Path(DATA_DIR) / f'{panel_id}_moments.pkl'
        if not moments_file.exists():
            print(f"      [FAIL] Moments file not found: {moments_file}")
            stats_passed = False
        else:
            with open(moments_file, 'rb') as f:
                moments_data = pickle.load(f)

            # Test three months: first, last, and random middle
            min_month = dkkm_stats['month'].min()
            max_month = dkkm_stats['month'].max()
            random_month = random.randint(min_month + 1, max_month - 1)
            test_months = [min_month, random_month, max_month]
            # Get first alpha from stats (usually 0)
            test_alpha = dkkm_stats['alpha'].iloc[0]
            # Get matrix index (usually 0 for single matrix)
            matrix_idx = dkkm_stats['matrix'].iloc[0]

            print(f"      Testing 3 months: {min_month} (first), {random_month} (random), {max_month} (last)")
            print(f"      Alpha={test_alpha}, Matrix={matrix_idx}")

            # Loop through all three test months
            for test_month in test_months:
                print(f"\n      Month {test_month}:")

                # Get stats from pickle
                dkkm_row = dkkm_stats[(dkkm_stats['month'] == test_month) &
                                      (dkkm_stats['alpha'] == test_alpha) &
                                      (dkkm_stats['matrix'] == matrix_idx)].iloc[0]

                include_mkt_test = bool(dkkm_row['include_mkt'])
                print(f"        include_mkt={include_mkt_test}, alpha={test_alpha}")

                # Get moments for this month (moments_data has nested structure)
                moments = moments_data['moments']
                month_moments = moments[test_month]
                cond_var = month_moments['cond_var']
                rp = month_moments['rp']
                second_moment = month_moments['second_moment']
                second_moment_inv = month_moments['second_moment_inv']

                # Get panel data for this month
                panel_data = panel.set_index(['month', 'firmid'])
                data_month = panel_data.loc[test_month]

                # Compute factor loadings using DKKM with RFF
                import dkkm_functions as dkkm_old
                if model == 'bgn':
                    rf = data_month['rf_stand']
                else:
                    rf = None

                factor_loadings, _ = dkkm_old.rff(data_month[chars], rf, W, model)

                # Get portfolio of factors (MV-efficient portfolio)
                # Use the factors computed with current code (respects rank_standardize flag)
                dkkm_returns = factors_new

                # Use original mve_data to match production code behavior
                # Pass nfeatures * alpha to match portfolio_stats.py line 306
                nfeatures = dkkm_returns.shape[1]
                port_of_factors_df = dkkm_old.mve_data(
                    f=dkkm_returns,
                    month=test_month,
                    alpha_lst=nfeatures * np.array([test_alpha]),
                    mkt_rf=None if not include_mkt_test else None  # Would need market returns if include_mkt
                )
                port_of_factors = port_of_factors_df.values.flatten()

                # Compute weights on stocks
                weights_partial = factor_loadings @ port_of_factors

                # Create full N-dimensional weight vector
                N_moments = moments_data['N']
                weights_on_stocks = np.zeros(N_moments)
                firm_ids = data_month.index.to_numpy()
                weights_on_stocks[firm_ids] = weights_partial

                # Manually compute stats using original formulas
                manual_stdev = np.sqrt(weights_on_stocks @ cond_var @ weights_on_stocks)
                manual_mean = weights_on_stocks @ rp
                manual_xret = weights_on_stocks @ data_month['xret'].values
                errs = rp - second_moment @ weights_on_stocks
                manual_hjd = np.sqrt(errs @ second_moment_inv @ errs)

                # Compare with pickle values
                print(f"        Comparing computed vs. pickle values:")
                try:
                    assert_close(manual_stdev, dkkm_row['stdev'], rtol=1e-10, atol=1e-12, name="stdev")
                    print(f"          [PASS] stdev matches")
                except AssertionError as e:
                    print(f"          [FAIL] stdev: {e}")
                    stats_passed = False

                try:
                    assert_close(manual_mean, dkkm_row['mean'], rtol=1e-10, atol=1e-12, name="mean")
                    print(f"          [PASS] mean matches")
                except AssertionError as e:
                    print(f"          [FAIL] mean: {e}")
                    stats_passed = False

                try:
                    assert_close(manual_xret, dkkm_row['xret'], rtol=1e-10, atol=1e-12, name="xret")
                    print(f"          [PASS] xret matches")
                except AssertionError as e:
                    print(f"          [FAIL] xret: {e}")
                    stats_passed = False

                try:
                    assert_close(manual_hjd, dkkm_row['hjd'], rtol=1e-10, atol=1e-12, name="hjd")
                    print(f"          [PASS] hjd matches")
                except AssertionError as e:
                    print(f"          [FAIL] hjd: {e}")
                    stats_passed = False

                # Test sdf_ret (should match moments file exactly)
                try:
                    expected_sdf_ret = month_moments['sdf_ret']
                    assert_close(expected_sdf_ret, dkkm_row['sdf_ret'], rtol=1e-14, atol=1e-15, name="sdf_ret")
                    print(f"          [PASS] sdf_ret matches")
                except AssertionError as e:
                    print(f"          [FAIL] sdf_ret: {e}")
                    stats_passed = False

    # Print stats verification summary
    if stats_passed:
        print(f"\n      [ALL PASS] Portfolio statistics verification passed")
    else:
        print(f"\n      [FAIL] Portfolio statistics verification failed")

    # Step 6: Compare original vs current factor returns
    print(f"\n[COMPARE] Comparing original vs current factor returns...")
    all_passed = True

    # Compare appropriate version based on rank_standardize flag
    if rank_standardize:
        factors_old = f_rs_old
        factor_name = "Rank-standardized returns"
    else:
        factors_old = f_nors_old
        factor_name = "Non-rank-standardized returns"

    print(f"\n  {factor_name}:")
    print(f"    Original shape: {factors_old.shape}")
    print(f"    Current shape: {factors_new.shape}")

    try:
        # Compare values
        assert_close(
            factors_new.values,
            factors_old.values,
            rtol=1e-14,
            atol=1e-15,
            name=factor_name
        )
        print(f"    [PASS] Values are identical")

        # Check index match
        assert factors_new.index.equals(factors_old.index), f"{factor_name} index mismatch"
        print(f"    [PASS] Index matches")

    except AssertionError as e:
        print(f"    [FAIL] {factor_name}: {e}")
        all_passed = False

    if all_passed:
        print("\n  [ALL PASS] All factor comparisons passed")
    else:
        print("\n  [FAIL] Factors are NOT identical")

    # Combine results
    overall_passed = all_passed and stats_passed

    # Save outputs to tests/outputs directory
    print(f"\n[SAVE] Saving DKKM results to tests/outputs...")
    import shutil
    outputs_dir = Path(__file__).parent / 'outputs'
    outputs_dir.mkdir(exist_ok=True)

    # Save new code results
    if dkkm_file.exists():
        dst_file = outputs_dir / f'{panel_id}_dkkm_{nfeatures}_new.pkl'
        shutil.copy2(dkkm_file, dst_file)
        print(f"      Saved new DKKM to {dst_file.name}")

    # Save original code results
    dkkm_original_data = {
        'factors_rs': f_rs_old,
        'factors_nors': f_nors_old,
        'W': W,
        'rank_standardize': rank_standardize
    }
    dkkm_original_file = outputs_dir / f'{panel_id}_dkkm_{nfeatures}_original.pkl'
    with open(dkkm_original_file, 'wb') as f:
        pickle.dump(dkkm_original_data, f)
    print(f"      Saved original DKKM to {dkkm_original_file.name}")

    # Cleanup - delete all pickle files from DATA_DIR
    print(f"\n[CLEANUP] Deleting pickle files from DATA_DIR...")

    files_to_delete = [dkkm_file]
    for file_path in files_to_delete:
        if file_path.exists():
            file_path.unlink()
            print(f"      Deleted {file_path.name}")
        else:
            print(f"      {file_path.name} not found (already deleted)")

    # Note: panel_file and moments_file are left for other tests to use/cleanup

    print("\n" + "=" * 70)
    print(f"[DONE] DKKM factor test complete for {model.upper()}")
    if overall_passed:
        print("[RESULT] ALL TESTS PASSED")
    else:
        print("[RESULT] SOME TESTS FAILED")
    print("=" * 70)

    return overall_passed


def main():
    """Main execution function."""
    # Check configuration values first
    if not check_config_values():
        sys.exit(1)

    if len(sys.argv) < 3:
        print("Usage: python test_dkkm.py <model> <nfeatures>")
        print("where:")
        print("  <model> is one of: bgn, kp14, gs21")
        print("  <nfeatures> is the number of random features (e.g., 6, 36, 360)")
        print("\nExamples:")
        print("  python test_dkkm.py bgn 360")
        print("  python test_dkkm.py kp14 36")
        sys.exit(1)

    model = sys.argv[1].lower()

    try:
        nfeatures = int(sys.argv[2])
    except ValueError:
        print(f"Error: nfeatures must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)

    if model not in ['bgn', 'kp14', 'gs21']:
        print(f"Error: Unknown model '{model}'")
        print("Valid models: bgn, kp14, gs21")
        sys.exit(1)

    if nfeatures <= 0:
        print(f"Error: nfeatures must be positive, got {nfeatures}")
        sys.exit(1)

    success = test_dkkm_factors(model, nfeatures)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
