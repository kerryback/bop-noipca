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
from .sdf_utils import load_precomputed_moments
from .fama_functions import fama_french

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

    DKKM-specific scaling: Applies nfeatures * alpha scaling internally.
    Combined with ridge_regression_grid's 360* scaling, gives final penalty: 360 * nfeatures * alpha

    Args:
        f: DataFrame of factor returns
        month: Current month
        alpha_lst: Array of BASE ridge penalties (unscaled from config)
        mkt_rf: Market return (if including market)

    Returns:
        DataFrame of portfolio weights (columns = original alpha values)
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

    # Number of features (for DKKM-specific penalty scaling)
    # Exclude market column from feature count if present
    nfeatures = X.shape[1] - 1 if include_mkt else X.shape[1]

    # Save original alphas for column names
    original_alphas = alpha_lst.copy()

    # Handle market (don't penalize last variable)
    if include_mkt:
        # For alpha=0, just solve directly
        beta_0 = ridge_regression_grid(X, y, np.array([0]))[:, 0]
        betas_list = [beta_0]
        result_alphas = [original_alphas[0]]  # Should be 0

        # For alpha > 0, augment design matrix to avoid penalizing market
        for i, alpha in enumerate(alpha_lst):
            if alpha > 0:
                # Augment: don't penalize last column
                # Scale by nfeatures to get effective penalty: 360 * nfeatures * alpha
                X_aug = np.vstack([
                    X,
                    np.sqrt(360 * nfeatures * alpha) * np.eye(X.shape[1])[:-1]
                ])
                y_aug = np.concatenate([y, np.zeros(X.shape[1] - 1)])

                beta = ridge_regression_grid(X_aug, y_aug, np.array([0]))[:, 0]
                betas_list.append(beta)
                result_alphas.append(original_alphas[i])

        # Stack coefficients
        betas = np.column_stack(betas_list)
        result_alphas = np.array(result_alphas)

    else:
        # Standard ridge regression for all alphas at once
        # Scale by nfeatures here (ridge_regression_grid will apply 360* internally)
        # Final penalty: 360 * nfeatures * alpha
        scaled_alphas = nfeatures * alpha_lst
        betas = ridge_regression_grid(X, y, scaled_alphas)
        result_alphas = original_alphas

    # Return as DataFrame (columns = original unscaled alpha values)
    return pd.DataFrame(betas, index=index_cols, columns=result_alphas)


def compute_portfolio_stats(
    dkkm_returns: pd.DataFrame,
    panel: pd.DataFrame,
    panel_id: str,
    model: str,
    W: np.ndarray,
    chars: list,
    start_month: int,
    end_month: int,
    alpha_lst: list,
    include_mkt: bool = True,
    mkt_returns: pd.DataFrame = None,
    matrix_idx: int = 0,
    burnin: int = None
) -> pd.DataFrame:
    """
    Compute portfolio statistics for DKKM factors.

    Evaluates across alpha grid.

    Args:
        dkkm_returns: DKKM factor returns DataFrame
        panel: Panel data
        panel_id: Panel identifier (e.g., 'kp14_0')
        model: Model name ('bgn', 'kp14', 'gs21')
        W: Random feature matrix
        chars: List of characteristics
        start_month: First month (must be >= burnin + 360)
        end_month: Last month
        alpha_lst: List of ridge penalties (REQUIRED - no default)
        include_mkt: Whether market factor is included
        mkt_returns: Market factor returns (if include_mkt=True)
        matrix_idx: Random matrix index (for tracking)
        burnin: Burn-in period (must be from config.BGN_BURNIN/KP14_BURNIN/GS21_BURNIN)

    Returns:
        DataFrame with columns: ['month', 'matrix', 'alpha', 'include_mkt', 'stdev', 'mean', 'xret', 'sdf_ret', 'hjd']
    """
    if alpha_lst is None:
        raise ValueError("alpha_lst cannot be None - must provide explicit list of ridge penalties")

    if include_mkt and mkt_returns is None:
        raise ValueError(
            "include_mkt=True but mkt_returns is None. "
            "Either provide mkt_returns (FF market returns) or set include_mkt=False."
        )

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

    # Get number of DKKM features (D)
    nfeatures = dkkm_returns.shape[1]

    for month in range(start_month, end_month + 1):
        # Get pre-computed SDF outputs for this month
        if month not in moments:
            raise KeyError(f"Month {month} not found in pre-computed moments")

        month_moments = moments[month]
        rp_full = month_moments['rp']
        cond_var_full = month_moments['cond_var']
        sdf_ret = month_moments['sdf_ret']

        for alpha in alpha_lst:
            # Pass unscaled alpha - mve_data handles all scaling internally
            mkt_rf = mkt_returns if include_mkt else None
            port_of_factors = mve_data(
                dkkm_returns,
                month,
                np.array([alpha]),  # Unscaled - mve_data applies 360*nfeatures scaling
                mkt_rf
            )

            # Get factor loadings using RFF
            data_month = panel.loc[month].copy()

            # Get risk-free rate for BGN model
            if model == 'bgn':
                rf = data_month['rf_stand']
            else:
                rf = None

            # Compute RFF features (loadings)
            loadings, _ = rff(data_month[chars], rf, W, model)

            # Add market weights when include_mkt=True AND mkt_returns is available
            # This matches root code: factor_weights['mkt_rf'] = fama.fama_french(...)[:, -1]
            if include_mkt and mkt_rf is not None:
                mkt_weights = fama_french(data_month[chars], chars, data_month['mve'])[:, -1]
                loadings['mkt_rf'] = mkt_weights

            # Portfolio weights on stocks (partial - only for present firms)
            weights_on_stocks = loadings.values @ port_of_factors.values.flatten()

            # Get firm IDs from data_month (firms present at this month)
            firm_ids = data_month.index.to_numpy() if isinstance(data_month.index, pd.Index) else data_month['firmid'].to_numpy()

            # CRITICAL: Subset matrices to only firms present at this month
            # This matches root code behavior where matrices are subsetted before computation
            rp = rp_full[firm_ids]
            cond_var = cond_var_full[firm_ids, :][:, firm_ids]
            second_moment = cond_var + np.outer(rp, rp)
            second_moment_inv = np.linalg.pinv(second_moment)

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
                'matrix': matrix_idx,
                'alpha': alpha,
                'include_mkt': include_mkt,
                'stdev': stdev,
                'mean': mean,
                'xret': xret,
                'sdf_ret': sdf_ret,
                'hjd': hjd
            })

    return pd.DataFrame(results_list)
