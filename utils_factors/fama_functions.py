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
from .sdf_utils import load_precomputed_moments


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

    NOTE: For Fama methods, alpha should always be 0 (OLS, no penalization).

    Args:
        f: DataFrame of factor returns
        month: Current month
        alpha: Ridge penalty - should be 0 for Fama (OLS only)

    Returns:
        Portfolio weights as Series
    """
    # Use past 360 months of factor returns
    X = f.loc[month - 360:month - 1].dropna().to_numpy()
    y = np.ones(len(X))

    # NOTE: alpha should be 0 for Fama methods (OLS only)
    # With alpha=0, this reduces to OLS: (X'X)^{-1} X'y
    pi = ridge_regression_fast(X, y, alpha=360 * alpha)

    return pd.Series(pi, index=f.columns)


def compute_portfolio_stats(
    ff_returns: pd.DataFrame,
    fm_returns: pd.DataFrame,
    panel: pd.DataFrame,
    panel_id: str,
    model: str,
    chars: list,
    start_month: int,
    end_month: int,
    alpha_lst: list = None,  # IGNORED - Fama methods use OLS only (alpha=0)
    burnin: int = None
) -> pd.DataFrame:
    """
    Compute portfolio statistics for Fama factors (FF and FM).

    NOTE: Penalization disabled - uses OLS only (alpha=0).

    Args:
        ff_returns: Fama-French factor returns DataFrame
        fm_returns: Fama-MacBeth factor returns DataFrame
        panel: Panel data
        panel_id: Panel identifier (e.g., 'kp14_0')
        model: Model name ('bgn', 'kp14', 'gs21')
        chars: List of characteristics
        start_month: First month (must be >= burnin + 360)
        end_month: Last month
        alpha_lst: IGNORED - kept for API compatibility
        burnin: Burn-in period (must be from config.BGN_BURNIN/KP14_BURNIN/GS21_BURNIN)

    Returns:
        DataFrame with columns: ['month', 'method', 'alpha', 'stdev', 'mean', 'xret', 'sdf_ret', 'hjd']
    """
    # PENALIZATION COMMENTED OUT - USE OLS ONLY
    # if alpha_lst is None:
    #     alpha_lst = [0]

    # Force alpha=0 (OLS) for Fama methods
    alpha = 0

    # Load pre-computed SDF moments
    moments, N, moments_start, moments_end = load_precomputed_moments(panel_id)

    # Clamp start_month to moments range (portfolio stats need pre-computed moments)
    if start_month < moments_start:
        print(f"  [INFO] Clamping start_month from {start_month} to {moments_start} (moments range)")
        start_month = moments_start

    if end_month > moments_end:
        raise ValueError(
            f"Requested end_month {end_month} exceeds available moments range "
            f"[{moments_start}, {moments_end}] for panel {panel_id}. "
            f"Recompute moments with a wider range."
        )

    results_list = []

    # Combine FF and FM returns
    fama_methods = {
        'ff': ff_returns,
        'fm': fm_returns
    }

    for method_name, factor_returns in fama_methods.items():
        for month in range(start_month, end_month + 1):
            # Get pre-computed SDF outputs for this month
            if month not in moments:
                raise KeyError(f"Month {month} not found in pre-computed moments")

            month_moments = moments[month]
            rp_full = month_moments['rp']
            cond_var_full = month_moments['cond_var']
            sdf_ret = month_moments['sdf_ret']

            # PENALIZATION COMMENTED OUT - SINGLE ALPHA ONLY
            # for alpha in alpha_lst:
            # Use OLS (alpha=0) for portfolio optimization
            port_of_factors = mve_data(factor_returns, month, alpha)

            # Get factor loadings from panel
            # Recompute Fama factors for this month to get loadings
            data_month = panel.loc[month].copy()

            if method_name == 'ff':
                # Fama-French: Get loadings from factor construction
                loadings = fama_french(data_month, chars, data_month['mve'])
            else:
                # Fama-MacBeth: Get loadings from characteristics
                loadings = fama_macbeth(data_month, chars, stdz_fm=True)

            # Portfolio weights on stocks (only for present firms)
            weights_on_stocks = loadings @ port_of_factors.values

            # Get firm IDs from data_month (firms present at this month)
            firm_ids = data_month.index.to_numpy() if isinstance(data_month.index, pd.Index) else data_month['firmid'].to_numpy()

            # CRITICAL: Subset matrices to only firms present at this month
            # This matches root code behavior where matrices are subsetted before computation
            rp = rp_full[firm_ids]
            cond_var = cond_var_full[firm_ids, :][:, firm_ids]
            second_moment = cond_var + np.outer(rp, rp)
            second_moment_inv = linalg.pinv(second_moment)

            # Compute statistics using subsetted matrices
            stdev = np.sqrt(weights_on_stocks @ cond_var @ weights_on_stocks)
            mean = weights_on_stocks @ rp

            # Realized return
            xret = weights_on_stocks @ data_month['xret'].values

            # Hansen-Jagannathan distance (now computed correctly with subsetted matrices)
            errs = rp - second_moment @ weights_on_stocks
            hjd = np.sqrt(errs @ second_moment_inv @ errs)

            results_list.append({
                'month': month,
                'method': method_name,
                'alpha': alpha,  # Will always be 0 (OLS)
                'stdev': stdev,
                'mean': mean,
                'xret': xret,
                'sdf_ret': sdf_ret,
                'hjd': hjd
            })

    return pd.DataFrame(results_list)
