"""
Comprehensive test for DKKM mve_data fix.

This test verifies that the fix for nfeatures counting (excluding market column)
makes the refactored code produce identical results to the original code.

USAGE:
------
Run from tests/ directory:
    cd tests
    python test_dkkm_mve_fix.py

PURPOSE:
--------
Tests the specific bug fix where nfeatures was incorrectly including the market
column when computing ridge penalty scaling.

TESTS:
------
1. Test include_mkt=False (should have always been correct)
2. Test include_mkt=True (this is where the bug was)
3. Test multiple alpha values
4. Test multiple feature counts
5. Verify numerical identity (rtol=1e-14)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))  # ../ (noipca/)
sys.path.insert(0, str(Path(__file__).parent / 'original_code'))  # ./original_code/
sys.path.insert(0, str(Path(__file__).parent / 'test_utils'))  # ./test_utils/

from comparison import assert_close

# Import original code
import dkkm_functions as dkkm_old

# Import refactored code
from utils_factors import dkkm_functions as dkkm_new

# Test configuration
TEST_SEED = 12345
np.random.seed(TEST_SEED)


def generate_test_data(n_months=400, n_features=36):
    """Generate synthetic DKKM factor returns for testing."""
    # Generate random factor returns
    factor_returns = pd.DataFrame(
        np.random.randn(n_months, n_features) * 0.02 + 0.001,
        index=range(n_months),
        columns=[str(i) for i in range(n_features)]
    )

    # Generate market returns
    market_returns = pd.Series(
        np.random.randn(n_months) * 0.04 + 0.005,
        index=range(n_months),
        name='mkt_rf'
    )

    return factor_returns, market_returns


def test_mve_data_no_market():
    """Test mve_data WITHOUT market (should have always worked)."""
    print("\n" + "="*70)
    print("TEST 1: mve_data WITHOUT market (include_mkt=False)")
    print("="*70)

    # Generate test data
    n_features = 36
    factor_returns, market_returns = generate_test_data(n_features=n_features)

    # Test parameters
    test_month = 380
    alpha_lst = np.array([0.0, 0.01, 0.05, 0.1])

    print(f"Testing with {n_features} features, {len(alpha_lst)} alpha values")
    print(f"Alpha values: {alpha_lst}")

    # Original code: expects alpha_lst to be PRE-SCALED by nfeatures
    scaled_alphas_old = n_features * alpha_lst
    port_old = dkkm_old.mve_data(
        factor_returns,
        test_month,
        scaled_alphas_old,
        mkt_rf=None
    )

    # Refactored code: expects UNSCALED alpha_lst (handles scaling internally)
    port_new = dkkm_new.mve_data(
        factor_returns,
        test_month,
        alpha_lst,
        mkt_rf=None
    )

    print(f"\nOriginal portfolio shape: {port_old.shape}")
    print(f"Refactored portfolio shape: {port_new.shape}")

    # Compare results
    all_passed = True
    for i, alpha in enumerate(alpha_lst):
        try:
            # Original uses scaled alpha as column name
            col_old = scaled_alphas_old[i]
            # Refactored uses unscaled alpha as column name
            col_new = alpha

            assert_close(
                port_new[col_new].values,
                port_old[col_old].values,
                rtol=1e-14,
                atol=1e-15,
                name=f"alpha={alpha}"
            )
            print(f"  [PASS] alpha={alpha}")
        except AssertionError as e:
            print(f"  [FAIL] alpha={alpha}: {e}")
            all_passed = False

    if all_passed:
        print("\n[PASS] TEST 1 PASSED: No market case works correctly")
    else:
        print("\n[FAIL] TEST 1 FAILED: No market case has discrepancies")

    return all_passed


def test_mve_data_with_market():
    """Test mve_data WITH market (this is where the bug was)."""
    print("\n" + "="*70)
    print("TEST 2: mve_data WITH market (include_mkt=True) - THE BUG FIX")
    print("="*70)

    # Generate test data
    n_features = 36
    factor_returns, market_returns = generate_test_data(n_features=n_features)

    # Test parameters
    test_month = 380
    alpha_lst = np.array([0.0, 0.01, 0.05, 0.1])

    print(f"Testing with {n_features} features + market, {len(alpha_lst)} alpha values")
    print(f"Alpha values: {alpha_lst}")
    print(f"\nCRITICAL: Original code scales penalty by {n_features}")
    print(f"          Refactored (fixed) should also scale by {n_features} (NOT {n_features+1})")

    # Original code: expects alpha_lst to be PRE-SCALED by nfeatures
    scaled_alphas_old = n_features * alpha_lst
    port_old = dkkm_old.mve_data(
        factor_returns,
        test_month,
        scaled_alphas_old,
        mkt_rf=market_returns
    )

    # Refactored code: expects UNSCALED alpha_lst (handles scaling internally)
    port_new = dkkm_new.mve_data(
        factor_returns,
        test_month,
        alpha_lst,
        mkt_rf=market_returns
    )

    print(f"\nOriginal portfolio shape: {port_old.shape}")
    print(f"Refactored portfolio shape: {port_new.shape}")

    # Compare results
    all_passed = True
    for i, alpha in enumerate(alpha_lst):
        try:
            # Original uses scaled alpha as column name
            col_old = scaled_alphas_old[i]
            # Refactored uses unscaled alpha as column name
            col_new = alpha

            # Get portfolio weights
            weights_old = port_old[col_old].values
            weights_new = port_new[col_new].values

            # Check that shapes match (should include market)
            assert len(weights_old) == n_features + 1, f"Original should have {n_features}+1 weights"
            assert len(weights_new) == n_features + 1, f"Refactored should have {n_features}+1 weights"

            # Check numerical identity
            assert_close(
                weights_new,
                weights_old,
                rtol=1e-14,
                atol=1e-15,
                name=f"alpha={alpha}"
            )

            # Also check that market weight is present and last
            print(f"  [PASS] alpha={alpha}, market_weight_old={weights_old[-1]:.6f}, market_weight_new={weights_new[-1]:.6f}")

        except AssertionError as e:
            print(f"  [FAIL] alpha={alpha}: {e}")

            # Print diagnostic info
            weights_old = port_old[col_old].values
            weights_new = port_new[col_new].values

            max_diff = np.max(np.abs(weights_old - weights_new))
            mean_diff = np.mean(np.abs(weights_old - weights_new))
            print(f"         Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
            print(f"         Sample weights (first 5):")
            print(f"         Original:   {weights_old[:5]}")
            print(f"         Refactored: {weights_new[:5]}")

            all_passed = False

    if all_passed:
        print("\n[PASS] TEST 2 PASSED: Market case now works correctly after fix!")
    else:
        print("\n[FAIL] TEST 2 FAILED: Market case still has discrepancies")
        print("\nThis means the bug fix did not fully resolve the issue.")
        print("The problem may be elsewhere in the code.")

    return all_passed


def test_different_feature_counts():
    """Test with different numbers of features to verify scaling."""
    print("\n" + "="*70)
    print("TEST 3: Different feature counts")
    print("="*70)

    feature_counts = [6, 18, 36, 180]
    test_month = 380
    alpha_lst = np.array([0.0, 0.01])  # Use multiple alphas to avoid original code bug

    all_passed = True

    for n_features in feature_counts:
        print(f"\nTesting with {n_features} features...")

        # Generate test data
        factor_returns, market_returns = generate_test_data(n_features=n_features)

        # Test WITHOUT market
        scaled_alphas_old = n_features * alpha_lst
        port_old_nomkt = dkkm_old.mve_data(
            factor_returns, test_month, scaled_alphas_old, mkt_rf=None
        )
        port_new_nomkt = dkkm_new.mve_data(
            factor_returns, test_month, alpha_lst, mkt_rf=None
        )

        # Test WITH market
        port_old_mkt = dkkm_old.mve_data(
            factor_returns, test_month, scaled_alphas_old, mkt_rf=market_returns
        )
        port_new_mkt = dkkm_new.mve_data(
            factor_returns, test_month, alpha_lst, mkt_rf=market_returns
        )

        try:
            # Check both alphas
            for i, alpha in enumerate(alpha_lst):
                scaled_alpha = scaled_alphas_old[i]

                # Check no market case
                assert_close(
                    port_new_nomkt[alpha].values,
                    port_old_nomkt[scaled_alpha].values,
                    rtol=1e-14,
                    atol=1e-15,
                    name=f"n_features={n_features}, no_market, alpha={alpha}"
                )

                # Check with market case (THE CRITICAL TEST)
                assert_close(
                    port_new_mkt[alpha].values,
                    port_old_mkt[scaled_alpha].values,
                    rtol=1e-14,
                    atol=1e-15,
                    name=f"n_features={n_features}, with_market, alpha={alpha}"
                )

            print(f"  [PASS] {n_features} features: both cases match for all alphas")

        except AssertionError as e:
            print(f"  [FAIL] {n_features} features: {e}")
            all_passed = False

    if all_passed:
        print("\n[PASS] TEST 3 PASSED: All feature counts work correctly")
    else:
        print("\n[FAIL] TEST 3 FAILED: Some feature counts have issues")

    return all_passed


def test_penalty_scaling_verification():
    """Verify that the effective penalty scaling is correct."""
    print("\n" + "="*70)
    print("TEST 4: Penalty scaling verification")
    print("="*70)

    n_features = 36
    factor_returns, market_returns = generate_test_data(n_features=n_features)
    test_month = 380

    # Test with multiple alphas (avoid original code bug)
    alpha_lst = np.array([0.0, 0.01])
    alpha = alpha_lst[1]  # Use alpha=0.01 for display

    print(f"\nWith {n_features} features and alpha={alpha}:")
    print(f"Expected effective penalty: 360 x {n_features} x {alpha} = {360 * n_features * alpha}")

    # Get portfolios
    scaled_alphas_old = n_features * alpha_lst

    # Without market
    port_old_nomkt = dkkm_old.mve_data(
        factor_returns, test_month, scaled_alphas_old, mkt_rf=None
    )
    port_new_nomkt = dkkm_new.mve_data(
        factor_returns, test_month, alpha_lst, mkt_rf=None
    )

    # With market
    port_old_mkt = dkkm_old.mve_data(
        factor_returns, test_month, scaled_alphas_old, mkt_rf=market_returns
    )
    port_new_mkt = dkkm_new.mve_data(
        factor_returns, test_month, alpha_lst, mkt_rf=market_returns
    )

    # Check L2 norms (should be similar for same penalty)
    scaled_alpha_display = scaled_alphas_old[1]  # Use index 1 (alpha=0.01)
    alpha_display = alpha_lst[1]

    norm_old_nomkt = np.linalg.norm(port_old_nomkt[scaled_alpha_display].values)
    norm_new_nomkt = np.linalg.norm(port_new_nomkt[alpha_display].values)
    norm_old_mkt = np.linalg.norm(port_old_mkt[scaled_alpha_display].values[:-1])  # Exclude market
    norm_new_mkt = np.linalg.norm(port_new_mkt[alpha_display].values[:-1])  # Exclude market

    print(f"\nL2 norm of portfolio weights (excluding market):")
    print(f"  No market:   Original={norm_old_nomkt:.6f}, Refactored={norm_new_nomkt:.6f}")
    print(f"  With market: Original={norm_old_mkt:.6f}, Refactored={norm_new_mkt:.6f}")

    # If the fix is correct, the norms should be very close
    all_passed = True
    try:
        assert_close(
            np.array([norm_new_nomkt]),
            np.array([norm_old_nomkt]),
            rtol=1e-12,
            atol=1e-14,
            name="L2 norm (no market)"
        )
        assert_close(
            np.array([norm_new_mkt]),
            np.array([norm_old_mkt]),
            rtol=1e-12,
            atol=1e-14,
            name="L2 norm (with market)"
        )
        print("\n[PASS] TEST 4 PASSED: Penalty scaling is correct")
    except AssertionError as e:
        print(f"\n[FAIL] TEST 4 FAILED: {e}")
        all_passed = False

    return all_passed


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("COMPREHENSIVE DKKM mve_data FIX TEST")
    print("="*70)
    print(f"Testing the fix: nfeatures = X.shape[1] - 1 if include_mkt else X.shape[1]")
    print(f"Random seed: {TEST_SEED}")

    results = []

    # Run all tests
    results.append(("No market case", test_mve_data_no_market()))
    results.append(("With market case (BUG FIX)", test_mve_data_with_market()))
    results.append(("Different feature counts", test_different_feature_counts()))
    results.append(("Penalty scaling", test_penalty_scaling_verification()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n" + "="*70)
        print("*** ALL TESTS PASSED! ***")
        print("="*70)
        print("\nThe fix correctly resolves the nfeatures counting bug.")
        print("Original and refactored code now produce identical results.")
        return 0
    else:
        print("\n" + "="*70)
        print("*** SOME TESTS FAILED ***")
        print("="*70)
        print("\nThere may be additional issues beyond the nfeatures counting bug.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
