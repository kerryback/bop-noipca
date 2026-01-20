# DKKM KP14 Bug Fix Summary

**Date:** 2026-01-19
**Issue:** Different results between original and refactored code for DKKM method in KP14 model
**Status:** ✓ FIXED AND VERIFIED

---

## The Problem

When `include_mkt=True`, the refactored code was incorrectly including the market column in the feature count used for penalty scaling:

```python
# BEFORE (INCORRECT):
nfeatures = X.shape[1]  # Includes market column when present
```

This caused:
- **Original code**: penalty = 360 × D × α (where D = number of DKKM features)
- **Refactored code**: penalty = 360 × (D+1) × α (incorrectly including market)

### Example Impact

With D=36 DKKM features and `include_mkt=True`:
- **Original**: penalty = 360 × 36 × α = 12,960 × α
- **Refactored (before fix)**: penalty = 360 × 37 × α = 13,320 × α
- **Difference**: 2.8% stronger penalty (~360 × α extra)

This resulted in:
- Different portfolio weights
- Lower volatility (over-penalized)
- Different Sharpe ratios
- Different HJD values

---

## The Fix

Changed line 202-203 in both files:
- `noipca/utils_factors/dkkm_functions.py`
- `reorg/utils_factors/dkkm_functions.py`

```python
# AFTER (CORRECT):
# Number of features (for DKKM-specific penalty scaling)
# Exclude market column from feature count if present
nfeatures = X.shape[1] - 1 if include_mkt else X.shape[1]
```

This ensures `nfeatures` always represents the number of DKKM features only, excluding the market column even when it's present in the design matrix.

---

## Verification

Created comprehensive test: `noipca/tests/test_dkkm_mve_fix.py`

### Test Results

All 4 test suites passed:

1. **[PASS] No market case** (include_mkt=False)
   - 4 alpha values: [0.0, 0.01, 0.05, 0.1]
   - All portfolios match exactly (rtol=1e-14)

2. **[PASS] With market case (THE BUG FIX)** (include_mkt=True)
   - 4 alpha values: [0.0, 0.01, 0.05, 0.1]
   - All portfolios match exactly (rtol=1e-14)
   - **This is where the bug was!**

3. **[PASS] Different feature counts**
   - Tested: 6, 18, 36, 180 features
   - Both include_mkt=True and False
   - All match across all configurations

4. **[PASS] Penalty scaling verification**
   - L2 norms match between original and refactored
   - Confirms effective penalty is identical

---

## Code Comparison

### Original Code (root/main.py + dkkm_functions.py)

```python
# main.py line 256
port_of_factors = dkkm.mve_data(
    frets.iloc[:, nf_indx],
    month,
    nfeatures * np.array(alpha_lst),  # ← Pre-scaled by nfeatures
    ff_rets.iloc[:, -1] if include_mkt else None
)

# dkkm_functions.py line 177-184
for alph in alpha_lst:  # alph = nfeatures * α_original
    if alph > 0:
        X_aug = np.concatenate((
            X,
            np.sqrt(360 * alph) * np.eye(X.shape[1])[:-1]  # ← Use alph directly
        ), axis=0)
```

**Effective penalty**: √(360 × nfeatures × α) ✓

### Refactored Code (After Fix)

```python
# portfolio_stats.py line 314
port_of_factors = dkkm.mve_data(
    dkkm_returns,
    month,
    np.array([alpha]),  # ← Unscaled - mve_data handles scaling
    mkt_rf
)

# dkkm_functions.py lines 202-203, 221
nfeatures = X.shape[1] - 1 if include_mkt else X.shape[1]  # ← FIXED

for i, alpha in enumerate(alpha_lst):  # alpha = α_original (unscaled)
    if alpha > 0:
        X_aug = np.vstack([
            X,
            np.sqrt(360 * nfeatures * alpha) * np.eye(X.shape[1])[:-1]  # ← Scale by nfeatures
        ])
```

**Effective penalty**: √(360 × nfeatures × α) ✓

---

## Mathematical Verification

Both implementations now produce:

**Ridge penalty** = 360 × nfeatures × α

Where:
- 360 = number of months in estimation window
- nfeatures = number of DKKM features (D), **excluding market**
- α = base ridge parameter from config

**When include_mkt=True:**
- nfeatures = D (number of DKKM columns in factor returns DataFrame)
- X.shape[1] = D + 1 (DKKM features + market)
- **Fix ensures**: nfeatures = X.shape[1] - 1 = D ✓

**When include_mkt=False:**
- nfeatures = D
- X.shape[1] = D
- **Fix ensures**: nfeatures = X.shape[1] = D ✓

---

## Impact Assessment

### Before Fix
- Portfolio weights differed by ~2-3% when include_mkt=True
- Sharpe ratios were systematically lower (over-penalized)
- Results did NOT match original implementation

### After Fix
- ✓ All portfolio weights match exactly (< 1e-14 relative error)
- ✓ All statistics (mean, stdev, HJD) match exactly
- ✓ Works correctly across all feature counts (6, 18, 36, 180)
- ✓ Works correctly with and without market

---

## Files Modified

1. `noipca/utils_factors/dkkm_functions.py` (line 203)
2. `reorg/utils_factors/dkkm_functions.py` (line 200)

Both files received identical fix.

---

## Testing

### To verify the fix:

```bash
cd noipca/tests
python test_dkkm_mve_fix.py
```

Expected output:
```
======================================================================
TEST SUMMARY
======================================================================
[PASS]: No market case
[PASS]: With market case (BUG FIX)
[PASS]: Different feature counts
[PASS]: Penalty scaling

======================================================================
*** ALL TESTS PASSED! ***
======================================================================
```

### To test full DKKM pipeline (when scripts are available):

```bash
cd noipca/tests
python test_dkkm.py kp14 36
```

---

## Conclusion

The bug has been **successfully fixed and verified**. The refactored code now produces numerically identical results to the original implementation across all test cases.

**Key lesson**: When augmenting design matrices with additional variables (like market returns), ensure that penalty scaling parameters reference the correct feature count, excluding any unpenalized variables.

---

## Additional Notes

### Original Code Bug

The original code (`root/dkkm_functions.py`) has a minor bug when `include_mkt=True` and `alpha_lst` is a single-element array - it fails to properly construct the output DataFrame. This doesn't affect production usage (which uses multiple alphas) but required the test to use at least 2 alpha values.

### HJD Computation

Unrelated to this fix: The refactored code correctly computes HJD with `np.sqrt()`, while the original code computes HJD² without the square root. This is documented separately in `CODE_COMPARISON.md`.
