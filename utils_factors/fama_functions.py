"""
Optimized Fama-French and Fama-MacBeth factor computation.

Key improvements:
- Vectorized operations where possible
- Reduced memory allocations
- Cleaner, more readable code
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import scipy.linalg as linalg
from typing import Callable, List, Tuple
from .ridge_utils import ridge_regression_fast
from .factor_utils import standardize_columns


def fama_french(
    data: pd.DataFrame,
    chars: List[str],
    mve: pd.Series,
    **kwargs
) -> np.ndarray:
    """
    Compute Fama-French factor portfolios.

    Constructs long-short portfolios based on 2x3 sorts on size and characteristics.

    Args:
        data: DataFrame with characteristics
        chars: List of characteristic names
        mve: Market value of equity (for value-weighting)
        **kwargs: Accepts additional arguments (e.g., stdz_fm) for compatibility

    Returns:
        weights: (N, K+1) array with factor weights + market
    """
    N = len(data)
    char_names = {
        'bm': 'hml',
        'agr': 'cma',
        'roe': 'rmw',
        'mom': 'umd'
    }

    # Get characteristic names for output
    if len(chars) == 3:
        names = ["smb", "hml", "umd"]
    else:
        names = ["smb", "hml", "cma", "rmw", "umd"]

    factor_dict = {}

    # Sort on size
    size_median = data["size"].median()
    big = (data["size"] > size_median).astype(float)
    small = 1 - big

    for char in chars:
        if char == "size":
            continue

        # Sort on characteristic (30/40/30 breakpoints)
        low = (data[char] <= data[char].quantile(0.3)).astype(float)
        high = (data[char] > data[char].quantile(0.7)).astype(float)
        med = 1 - low - high

        # Form six portfolios (value-weighted)
        portfolios = {
            'high_big': mve * high * big,
            'high_small': mve * high * small,
            'low_big': mve * low * big,
            'low_small': mve * low * small,
            'med_big': mve * med * big,
            'med_small': mve * med * small
        }

        # Normalize by portfolio market cap
        for key in portfolios:
            total_mve = portfolios[key].sum()
            if total_mve > 0:
                portfolios[key] /= total_mve

        # Construct long-short factor
        factor = 0.5 * (
            portfolios['high_big'] + portfolios['high_small']
            - portfolios['low_big'] - portfolios['low_small']
        )

        # Get standardized name
        factor_name = char_names.get(char, char)
        factor_dict[factor_name] = factor.to_numpy()

        # Define SMB using book-to-market terciles
        if char == "bm":
            smb = (
                portfolios['high_small'] + portfolios['med_small'] + portfolios['low_small']
                - portfolios['high_big'] - portfolios['med_big'] - portfolios['low_big']
            ) / 3
            factor_dict["smb"] = smb.to_numpy()

    # Create output DataFrame
    df = pd.DataFrame(factor_dict, index=data.index)

    # Flip sign for CMA (low minus high)
    if "cma" in df.columns:
        df["cma"] *= -1

    # Add value-weighted market portfolio
    df['mkt_rf'] = (mve / mve.sum()).to_numpy()

    return df.to_numpy()


def fama_macbeth(
    data: pd.DataFrame,
    chars: List[str],
    stdz_fm: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Compute Fama-MacBeth factor portfolios.

    Cross-sectional regression weights based on characteristics.

    Args:
        data: DataFrame with characteristics
        chars: List of characteristic names
        stdz_fm: If True, standardize characteristics (subtract mean, divide by std).
                 If False, use raw characteristics (matches original code).

    Returns:
        weights: (N, K+1) array with factor weights + market
    """
    # Drop NaN values
    d = data.dropna()
    N_full = len(data)
    N = len(d)

    # Get characteristic names - use actual chars list
    names = chars

    # Optionally standardize characteristics based on flag
    X = d[chars].to_numpy()
    if stdz_fm:
        X = standardize_columns(X)

    # Add intercept
    X = np.column_stack([np.ones(N), X])

    # Pseudo-inverse: P = X (X'X)^{-1}
    XTX_inv = linalg.pinvh(X.T @ X)
    P = X @ XTX_inv

    # Drop intercept column
    P = P[:, 1:]

    # Normalize to long-short (sum of absolute weights = 2)
    abs_sum = np.abs(P).sum(axis=0)
    abs_sum[abs_sum < 1e-10] = 1.0  # Avoid division by zero
    P = 2 * P / abs_sum

    # Add equal-weighted market portfolio column
    mkt_weights = np.ones((N, 1)) / N_full
    P_with_mkt = np.column_stack([P, mkt_weights])

    # Create output array for full data (including NaNs)
    result = np.zeros((N_full, len(names) + 1))

    # Get positional indices of non-NaN rows
    valid_positions = data.index.get_indexer(d.index)
    result[valid_positions, :] = P_with_mkt

    # Fill market weights for NaN rows too
    result[:, -1] = 1.0 / N_full

    return result


def factors(
    method: Callable,
    panel: pd.DataFrame,
    n_jobs: int,
    start: int,
    end: int,
    chars: List[str],
    stdz_fm: bool = False
) -> pd.DataFrame:
    """
    Compute panel of factor returns for a given method.

    Args:
        method: Function to compute factor weights (fama_french or fama_macbeth)
        panel: Panel data with multi-index (month, firmid)
        n_jobs: Number of parallel jobs
        start: Start month
        end: End month
        chars: List of characteristics
        stdz_fm: If True, standardize characteristics in Fama-MacBeth (ignored for Fama-French)

    Returns:
        Factor returns as DataFrame indexed by month
    """
    def monthly_rets(month: int) -> pd.DataFrame:
        """Compute factor returns for a single month."""
        data = panel.loc[month]
        weights = method(data[chars], chars, mve=data.mve, stdz_fm=stdz_fm)
        rets = data.xret.to_numpy().reshape(-1, 1)

        # Factor returns = weights' @ returns
        factor_rets = weights.T @ rets

        return pd.DataFrame(factor_rets.T, index=[month])

    # Parallel computation across months
    lst = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(monthly_rets)(month) for month in range(start, end + 1)
    )

    # Concatenate results
    result = pd.concat(lst)
    result.index.name = "month"

    return result


def mve_data(
    f: pd.DataFrame,
    month: int,
    alpha: float
) -> pd.Series:
    """
    Compute mean-variance efficient portfolio of factors.

    Uses ridge regression: argmin ||y - X*beta||^2 + alpha*||beta||^2
    where y = 1 (target return of 1) and X = factor returns.

    Args:
        f: DataFrame of factor returns
        month: Current month
        alpha: Ridge penalty (already scaled by caller if needed)

    Returns:
        Portfolio weights as Series
    """
    # Use past 360 months of factor returns
    X = f.loc[month - 360:month - 1].dropna().to_numpy()
    y = np.ones(len(X))

    # Caller is responsible for scaling alpha appropriately
    # (DKKM scales by nfeatures, Fama doesn't)
    pi = ridge_regression_fast(X, y, alpha=360 * alpha)

    return pd.Series(pi, index=f.columns)
