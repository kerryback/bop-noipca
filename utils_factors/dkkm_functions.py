"""
Optimized DKKM (Random Fourier Features) factor computation.

Key improvements:
- Cleaned up code structure
- Optimized ridge regression with grid
- Better memory management
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Tuple, List

# Try to import Numba-accelerated functions first
try:
    from .factor_utils_numba import rank_standardize
    from .dkkm_functions_numba import rff_compute_numba
    NUMBA_AVAILABLE = True
except ImportError:
    # Fall back to standard implementations
    NUMBA_AVAILABLE = False

# Import ridge regression
from .ridge_utils import ridge_regression_grid

# If Numba failed, import standard rank_standardize
if not NUMBA_AVAILABLE:
    from .factor_utils import rank_standardize

# Track whether we've printed the acceleration message
_ACCELERATION_PRINTED = False


def rff(
    data: pd.DataFrame,
    rf: pd.Series,
    W: np.ndarray,
    model: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Random Fourier Features for characteristics.

    Args:
        data: DataFrame with characteristics
        rf: Risk-free rate (only used for 'bgn' model)
        W: (D/2, L) weight matrix for RFF
        model: Model name ('bgn', 'kp14', 'gs21')

    Returns:
        rff_standardized: Rank-standardized RFF features
        rff_raw: Raw RFF features (before standardization)
    """
    # Rank-standardize characteristics
    X = rank_standardize(data.to_numpy())

    # Add risk-free rate for BGN model
    if model == 'bgn':
        # X is already a numpy array
        X_df = pd.DataFrame(X, index=data.index, columns=data.columns)
        X_df['rf'] = rf.values
        X = X_df.to_numpy()

    # Compute random features: [sin(W@X'), cos(W@X')]
    if NUMBA_AVAILABLE:
        # Use Numba-accelerated version (2-3x faster)
        arr = rff_compute_numba(W, X)  # (N, D)
    else:
        # Standard numpy implementation
        Z = W @ X.T  # (D/2, N)
        Z1 = np.sin(Z)  # (D/2, N)
        Z2 = np.cos(Z)  # (D/2, N)
        arr = np.vstack([Z1, Z2]).T  # (N, D)

    # Convert to DataFrame
    arr_df = pd.DataFrame(
        arr,
        index=data.index,
        columns=[str(i) for i in range(arr.shape[1])]
    )

    # Rank-standardize features
    arr_std = rank_standardize(arr)
    arr_std_df = pd.DataFrame(
        arr_std,
        index=data.index,
        columns=arr_df.columns
    )

    return arr_std_df, arr_df


def factors(
    panel: pd.DataFrame,
    W: np.ndarray,
    n_jobs: int,
    start: int,
    end: int,
    model: str,
    chars: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute panel of DKKM factor returns.

    Args:
        panel: Panel data with multi-index (month, firmid)
        W: (D/2, L) RFF weight matrix
        n_jobs: Number of parallel jobs
        start: Start month
        end: End month
        model: Model name
        chars: List of characteristic names

    Returns:
        f_rs: Factor returns (rank-standardized features)
        f_nors: Factor returns (non-rank-standardized features)
    """
    global _ACCELERATION_PRINTED
    if not _ACCELERATION_PRINTED:
        if NUMBA_AVAILABLE:
            print("[ACCELERATION] Using Numba-accelerated DKKM functions")
        else:
            print("[INFO] Using standard DKKM functions (Numba not available)")
        _ACCELERATION_PRINTED = True

    def monthly_rets(month: int) -> Tuple[int, pd.Series, pd.Series]:
        """Compute factor returns for a single month."""
        data = panel.loc[month]

        # Get risk-free rate for BGN model
        if model == 'bgn':
            rf = data.rf_stand
        else:
            rf = None

        # Compute RFF
        weights_rs, weights_nors = rff(data[chars], rf, W=W, model=model)

        # Factor returns = feature weights' @ excess returns
        f_rs = (weights_rs.T @ data.xret).astype(np.float32)
        f_nors = (weights_nors.T @ data.xret).astype(np.float32)

        return month, f_rs, f_nors

    # Parallel computation
    lst = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(monthly_rets)(month) for month in range(start, end + 1)
    )

    # Unpack results
    months = [x[0] for x in lst]
    f_rs_list = [x[1] for x in lst]
    f_nors_list = [x[2] for x in lst]

    # Create DataFrames
    f_rs = pd.concat(f_rs_list, axis=1).T
    f_rs["month"] = months
    f_rs.set_index("month", inplace=True)

    f_nors = pd.concat(f_nors_list, axis=1).T
    f_nors["month"] = months
    f_nors.set_index("month", inplace=True)

    return f_rs, f_nors


def mve_data(
    f: pd.DataFrame,
    month: int,
    alpha_lst: np.ndarray,
    mkt_rf: pd.Series = None
) -> pd.DataFrame:
    """
    Compute mean-variance efficient portfolios for grid of penalties.

    Args:
        f: DataFrame of factor returns
        month: Current month
        alpha_lst: Array of ridge penalties
        mkt_rf: Market return (if including market)

    Returns:
        DataFrame of portfolio weights (columns = different alphas)
    """
    # Get past 360 months
    X = f.loc[month - 360:month - 1].dropna().to_numpy()

    include_mkt = mkt_rf is not None

    # Add market if specified
    if include_mkt:
        mkt_data = mkt_rf.loc[month - 360:month - 1].dropna().to_numpy()
        X = np.column_stack((X, mkt_data))

    y = np.ones(len(X))
    index_cols = list(f.columns) + (['mkt_rf'] if include_mkt else [])

    # Number of features (for penalty scaling)
    nfeatures = X.shape[1]

    # Handle market (don't penalize last variable)
    if include_mkt:
        # For alpha=0, just solve directly
        beta_0 = ridge_regression_grid(X, y, np.array([0]))[:, 0]
        betas_list = [beta_0]

        # For alpha > 0, augment design matrix to avoid penalizing market
        for alpha in alpha_lst:
            if alpha > 0:
                # Augment: don't penalize last column
                # Penalty scales with nfeatures to match original code
                X_aug = np.vstack([
                    X,
                    np.sqrt(360 * nfeatures * alpha) * np.eye(X.shape[1])[:-1]
                ])
                y_aug = np.concatenate([y, np.zeros(X.shape[1] - 1)])

                beta = ridge_regression_grid(X_aug, y_aug, np.array([0]))[:, 0]
                betas_list.append(beta)

        # Stack coefficients
        betas = np.column_stack(betas_list)

    else:
        # Standard ridge regression for all alphas at once
        # Note: alpha_lst is already scaled by nfeatures in calling code (portfolio_stats.py)
        # ridge_regression_grid applies 360* internally, so just pass alpha_lst
        betas = ridge_regression_grid(X, y, alpha_lst)

    # Return as DataFrame
    return pd.DataFrame(betas, index=index_cols, columns=alpha_lst)
