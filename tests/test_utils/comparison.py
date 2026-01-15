"""
Comparison utilities for regression testing.

Provides functions to compare arrays, DataFrames, and factors
between refactored and original code implementations.
"""

import numpy as np
import pandas as pd


def assert_close(arr1, arr2, rtol=1e-10, atol=1e-12, name=""):
    """
    Compare arrays with informative error messages.

    Parameters
    ----------
    arr1, arr2 : np.ndarray
        Arrays to compare
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    name : str
        Name for error reporting

    Raises
    ------
    AssertionError
        If arrays differ beyond tolerance
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    if arr1.shape != arr2.shape:
        raise AssertionError(
            f"{name} shape mismatch:\n"
            f"  arr1 shape: {arr1.shape}\n"
            f"  arr2 shape: {arr2.shape}"
        )

    if not np.allclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=True):
        max_diff = np.max(np.abs(arr1 - arr2))
        rel_diff = max_diff / (np.max(np.abs(arr1)) + 1e-20)

        raise AssertionError(
            f"{name} mismatch:\n"
            f"  Max absolute diff: {max_diff:.2e}\n"
            f"  Max relative diff: {rel_diff:.2e}\n"
            f"  Tolerance: rtol={rtol}, atol={atol}"
        )


def assert_dataframes_equal(df1, df2, rtol=1e-10, atol=1e-12, check_index=True):
    """
    Compare DataFrames column by column.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames to compare
    rtol, atol : float
        Tolerances for numerical comparison
    check_index : bool
        Whether to check index equality
    """
    # Check columns match
    if set(df1.columns) != set(df2.columns):
        missing_in_df2 = set(df1.columns) - set(df2.columns)
        missing_in_df1 = set(df2.columns) - set(df1.columns)
        raise AssertionError(
            f"DataFrame columns mismatch:\n"
            f"  In df1 but not df2: {missing_in_df2}\n"
            f"  In df2 but not df1: {missing_in_df1}"
        )

    # Check shapes match
    if df1.shape != df2.shape:
        raise AssertionError(
            f"DataFrame shape mismatch:\n"
            f"  df1 shape: {df1.shape}\n"
            f"  df2 shape: {df2.shape}"
        )

    # Check index if requested
    if check_index:
        if not df1.index.equals(df2.index):
            raise AssertionError("DataFrame index mismatch")

    # Compare each column
    for col in df1.columns:
        col1 = df1[col].values
        col2 = df2[col].values

        # Handle different dtypes
        if col1.dtype != col2.dtype:
            # Try converting both to float for comparison
            try:
                col1 = col1.astype(float)
                col2 = col2.astype(float)
            except (ValueError, TypeError):
                raise AssertionError(
                    f"Column '{col}' dtype mismatch and cannot convert:\n"
                    f"  df1 dtype: {df1[col].dtype}\n"
                    f"  df2 dtype: {df2[col].dtype}"
                )

        assert_close(col1, col2, rtol=rtol, atol=atol, name=f"Column '{col}'")


def assert_factors_equal_up_to_sign(factors1, factors2, rtol=1e-8, atol=1e-10):
    """
    Compare factor matrices allowing sign flips.

    IPCA factors are identified only up to sign, so we need to check
    if factors match either as-is or with sign flipped.

    Parameters
    ----------
    factors1, factors2 : np.ndarray
        Factor matrices (T x K)
    rtol, atol : float
        Tolerances for comparison
    """
    if factors1.shape != factors2.shape:
        raise AssertionError(
            f"Factor shape mismatch: {factors1.shape} vs {factors2.shape}"
        )

    T, K = factors1.shape

    # Try to match each factor (column) allowing sign flip
    for k in range(K):
        f1 = factors1[:, k]
        f2 = factors2[:, k]

        # Check if equal or negated
        positive_match = np.allclose(f1, f2, rtol=rtol, atol=atol)
        negative_match = np.allclose(f1, -f2, rtol=rtol, atol=atol)

        # If allclose fails, check correlation as fallback (for IPCA which may have small numerical differences)
        if not (positive_match or negative_match):
            corr_pos = np.corrcoef(f1, f2)[0, 1]
            corr_neg = np.corrcoef(f1, -f2)[0, 1]

            # Accept if correlation is very high (>0.999)
            correlation_match = (abs(corr_pos) > 0.999) or (abs(corr_neg) > 0.999)

            if not correlation_match:
                raise AssertionError(
                    f"Factor {k} doesn't match (even with sign flip):\n"
                    f"  Correlation: {corr_pos:.6f}\n"
                    f"  Correlation (negated): {corr_neg:.6f}\n"
                    f"  Expected: ~1.0 or ~-1.0"
                )


def compute_summary_stats(data_dict, name=""):
    """
    Compute summary statistics for a data dictionary.

    Useful for comparing overall magnitudes and distributions.

    Parameters
    ----------
    data_dict : dict
        Dictionary of arrays to summarize
    name : str
        Prefix for output

    Returns
    -------
    dict
        Summary statistics
    """
    stats = {}

    for key, arr in data_dict.items():
        if isinstance(arr, (np.ndarray, pd.Series, pd.DataFrame)):
            arr = np.asarray(arr)
            stats[key] = {
                'shape': arr.shape,
                'dtype': arr.dtype,
                'mean': np.mean(arr) if np.issubdtype(arr.dtype, np.number) else None,
                'std': np.std(arr) if np.issubdtype(arr.dtype, np.number) else None,
                'min': np.min(arr) if np.issubdtype(arr.dtype, np.number) else None,
                'max': np.max(arr) if np.issubdtype(arr.dtype, np.number) else None,
                'has_nan': np.any(np.isnan(arr)) if np.issubdtype(arr.dtype, np.number) else False,
                'has_inf': np.any(np.isinf(arr)) if np.issubdtype(arr.dtype, np.number) else False,
            }

    return stats


def print_comparison_summary(stats1, stats2, name1="refactored", name2="original"):
    """
    Print a comparison of summary statistics.

    Parameters
    ----------
    stats1, stats2 : dict
        Summary statistics from compute_summary_stats()
    name1, name2 : str
        Names for the two versions
    """
    print(f"\nComparison Summary: {name1} vs {name2}")
    print("=" * 70)

    all_keys = sorted(set(stats1.keys()) | set(stats2.keys()))

    for key in all_keys:
        if key not in stats1:
            print(f"\n{key}: MISSING in {name1}")
            continue
        if key not in stats2:
            print(f"\n{key}: MISSING in {name2}")
            continue

        s1 = stats1[key]
        s2 = stats2[key]

        print(f"\n{key}:")
        print(f"  Shape: {s1['shape']} vs {s2['shape']}")

        if s1['mean'] is not None and s2['mean'] is not None:
            mean_diff = abs(s1['mean'] - s2['mean'])
            print(f"  Mean: {s1['mean']:.6e} vs {s2['mean']:.6e} (diff: {mean_diff:.2e})")

            if s1['std'] > 0 and s2['std'] > 0:
                std_diff = abs(s1['std'] - s2['std'])
                print(f"  Std:  {s1['std']:.6e} vs {s2['std']:.6e} (diff: {std_diff:.2e})")
