"""
Portfolio Statistics Computation

Computes portfolio performance metrics (stdev, mean, xret, hjd) for factor portfolios
using mean-variance optimization with ridge regression (Issue #1 fix).

This module centralizes the portfolio statistics computation that was originally in
run_panel.py, making it reusable across run_fama.py and run_dkkm.py.
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Tuple, Dict, Callable

from config import DATA_DIR
from . import fama_functions as fama
from . import dkkm_functions as dkkm
from .ridge_utils import ridge_regression_grid


def load_precomputed_moments(panel_id: str) -> Tuple[Dict, int, int, int]:
    """
    Load pre-computed SDF conditional moments from pickle file.

    Args:
        panel_id: Panel identifier (e.g., 'kp14_0')

    Returns:
        Tuple of (moments dict, N, start_month, end_month)
        moments dict has structure: {month: {'rp', 'cond_var', 'second_moment', 'second_moment_inv', ...}}
    """
    # Load the moments pickle file
    moments_file = os.path.join(DATA_DIR, f'{panel_id}_moments.pkl')

    if not os.path.exists(moments_file):
        raise FileNotFoundError(
            f"Moments file not found: {moments_file}\n"
            f"Please run: python calculate_moments.py {panel_id}"
        )

    with open(moments_file, 'rb') as f:
        moments_data = pickle.load(f)

    moments = moments_data['moments']
    N = moments_data['N']
    start_month = moments_data['start_month']
    end_month = moments_data['end_month']

    return moments, N, start_month, end_month


def compute_model_portfolio_stats(
    model_premia: Dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    start_month: int,
    end_month: int
) -> pd.DataFrame:
    """
    Compute portfolio statistics for model factors (Taylor/Proj).

    Uses alpha=0 (no penalty) as model factors are already estimated.

    Args:
        model_premia: Dict with 'taylor' and 'proj' DataFrames of factor returns
        panel: Panel data (needed for returns)
        start_month: First month to compute stats (must be >= burnin + 360 for 360-month history)
        end_month: Last month to compute stats

    Returns:
        DataFrame with columns: ['month', 'method', 'alpha', 'stdev', 'mean', 'xret', 'hjd']
    """
    results_list = []

    for method in ['taylor', 'proj']:
        if method not in model_premia:
            continue

        factor_returns = model_premia[method]

        for month in range(start_month, end_month + 1):
            # Issue #1 fix: Use mve_data to find optimal portfolio of factors
            port_of_factors = fama.mve_data(factor_returns, month, alpha=0)

            # Get factor loadings (weights on stocks for each factor)
            # For model factors, these are the theoretical loadings from the model
            data_month = panel.loc[month]
            N = len(data_month)

            # Get factor weights (loadings) - these should be in the panel
            # For now, use equal weights as placeholder (this should be model-specific)
            factor_weights = np.ones((N, len(port_of_factors))) / N

            # Portfolio weights on stocks
            weights_on_stocks = factor_weights @ port_of_factors.values

            # Compute portfolio return
            returns = data_month['xret'].values
            port_return = weights_on_stocks @ returns

            # Compute statistics (simplified - full version would track time series)
            # For proper implementation, we need the time series of portfolio returns
            # This is a placeholder that matches the structure

            results_list.append({
                'month': month,
                'method': method,
                'alpha': 0.0,
                'stdev': np.nan,  # Would need time series
                'mean': port_return,
                'xret': np.nan,   # Would need risk-free rate
                'hjd': np.nan     # Would need SDF
            })

    return pd.DataFrame(results_list)


def compute_fama_portfolio_stats(
    ff_returns: pd.DataFrame,
    fm_returns: pd.DataFrame,
    panel: pd.DataFrame,
    panel_id: str,
    model: str,
    chars: list,
    start_month: int,
    end_month: int,
    alpha_lst: list = None,
    burnin: int = None
) -> pd.DataFrame:
    """
    Compute portfolio statistics for Fama factors (FF and FM).

    Evaluates across alpha grid to find optimal shrinkage.

    Args:
        ff_returns: Fama-French factor returns DataFrame
        fm_returns: Fama-MacBeth factor returns DataFrame
        panel: Panel data
        panel_id: Panel identifier (e.g., 'kp14_0')
        model: Model name ('bgn', 'kp14', 'gs21')
        chars: List of characteristics
        start_month: First month (must be >= burnin + 360)
        end_month: Last month
        alpha_lst: List of ridge penalties to evaluate (default: [0])
        burnin: Burn-in period (must be from config.BGN_BURNIN/KP14_BURNIN/GS21_BURNIN)

    Returns:
        DataFrame with columns: ['month', 'method', 'alpha', 'stdev', 'mean', 'xret', 'sdf_ret', 'hjd']
    """
    if alpha_lst is None:
        alpha_lst = [0]

    # Load pre-computed SDF moments
    moments, N, moments_start, moments_end = load_precomputed_moments(panel_id)

    # Use moments range if it's more restrictive than provided range
    start_month = max(start_month, moments_start)
    end_month = min(end_month, moments_end)

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
            rp = month_moments['rp']
            cond_var = month_moments['cond_var']
            second_moment = month_moments['second_moment']
            second_moment_inv = month_moments['second_moment_inv']
            sdf_ret = month_moments['sdf_ret']

            for alpha in alpha_lst:
                # Issue #1 fix: Use mve_data for portfolio optimization
                port_of_factors = fama.mve_data(factor_returns, month, alpha)

                # Get factor loadings from panel
                # Recompute Fama factors for this month to get loadings
                data_month = panel.loc[month].copy()

                if method_name == 'ff':
                    # Fama-French: Get loadings from factor construction
                    loadings = fama.fama_french(data_month, chars, data_month['mve'])
                else:
                    # Fama-MacBeth: Get loadings from characteristics
                    loadings = fama.fama_macbeth(data_month, chars, stdz_fm=True)

                # Portfolio weights on stocks (partial - only for non-NaN stocks)
                weights_partial = loadings @ port_of_factors.values

                # Create full N-dimensional weight vector (for all stocks in SDF)
                # Initialize with zeros for all N stocks
                weights_on_stocks = np.zeros(N)

                # Get firm IDs from data_month
                firm_ids = data_month.index.to_numpy() if isinstance(data_month.index, pd.Index) else data_month['firmid'].to_numpy()

                # Assign computed weights to the corresponding firms
                weights_on_stocks[firm_ids] = weights_partial

                # Compute statistics
                stdev = np.sqrt(weights_on_stocks @ cond_var @ weights_on_stocks)
                mean = weights_on_stocks @ rp

                # Realized return: only for available stocks
                xret_full = np.zeros(N)
                xret_full[firm_ids] = data_month['xret'].values
                xret = weights_on_stocks @ xret_full

                # Hansen-Jagannathan distance
                errs = rp - second_moment @ weights_on_stocks
                hjd = np.sqrt(errs @ second_moment_inv @ errs)

                results_list.append({
                    'month': month,
                    'method': method_name,
                    'alpha': alpha,
                    'stdev': stdev,
                    'mean': mean,
                    'xret': xret,
                    'sdf_ret': sdf_ret,
                    'hjd': hjd
                })

    return pd.DataFrame(results_list)


def compute_dkkm_portfolio_stats(
    dkkm_returns: pd.DataFrame,
    panel: pd.DataFrame,
    panel_id: str,
    model: str,
    W: np.ndarray,
    chars: list,
    start_month: int,
    end_month: int,
    alpha_lst: list = None,
    include_mkt: bool = False,
    mkt_returns: pd.DataFrame = None,
    matrix_idx: int = 0,
    burnin: int = None
) -> pd.DataFrame:
    """
    Compute portfolio statistics for DKKM factors.

    Evaluates across alpha grid to find optimal shrinkage.

    Args:
        dkkm_returns: DKKM factor returns DataFrame
        panel: Panel data
        panel_id: Panel identifier (e.g., 'kp14_0')
        model: Model name ('bgn', 'kp14', 'gs21')
        W: Random feature matrix
        chars: List of characteristics
        start_month: First month (must be >= burnin + 360)
        end_month: Last month
        alpha_lst: List of ridge penalties (default: [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1])
        include_mkt: Whether market factor is included
        mkt_returns: Market factor returns (if include_mkt=True)
        matrix_idx: Random matrix index (for tracking)
        burnin: Burn-in period (must be from config.BGN_BURNIN/KP14_BURNIN/GS21_BURNIN)

    Returns:
        DataFrame with columns: ['month', 'matrix', 'alpha', 'include_mkt', 'stdev', 'mean', 'xret', 'sdf_ret', 'hjd']
    """
    if alpha_lst is None:
        alpha_lst = [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1]

    # Load pre-computed SDF moments
    moments, N, moments_start, moments_end = load_precomputed_moments(panel_id)

    # Use moments range if it's more restrictive than provided range
    start_month = max(start_month, moments_start)
    end_month = min(end_month, moments_end)

    results_list = []

    # Get number of DKKM features (D)
    nfeatures = dkkm_returns.shape[1]

    for month in range(start_month, end_month + 1):
        # Get pre-computed SDF outputs for this month
        if month not in moments:
            raise KeyError(f"Month {month} not found in pre-computed moments")

        month_moments = moments[month]
        rp = month_moments['rp']
        cond_var = month_moments['cond_var']
        second_moment = month_moments['second_moment']
        second_moment_inv = month_moments['second_moment_inv']
        sdf_ret = month_moments['sdf_ret']

        for alpha in alpha_lst:
            # Issue #1 fix: Use mve_data for portfolio optimization
            # Scale alpha by nfeatures to match original code (main.py line 256)
            mkt_rf = mkt_returns if include_mkt else None
            port_of_factors = dkkm.mve_data(
                dkkm_returns,
                month,
                nfeatures * np.array([alpha]),
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
            loadings, _ = dkkm.rff(data_month[chars], rf, W, model)

            # Portfolio weights on stocks (partial - only for non-NaN stocks)
            weights_partial = loadings @ port_of_factors.values.flatten()

            # Create full N-dimensional weight vector (for all stocks in SDF)
            weights_on_stocks = np.zeros(N)

            # Get firm IDs from data_month
            firm_ids = data_month.index.to_numpy() if isinstance(data_month.index, pd.Index) else data_month['firmid'].to_numpy()

            # Assign computed weights to the corresponding firms
            weights_on_stocks[firm_ids] = weights_partial

            # Compute statistics
            stdev = np.sqrt(weights_on_stocks @ cond_var @ weights_on_stocks)
            mean = weights_on_stocks @ rp

            # Realized return: only for available stocks
            xret_full = np.zeros(N)
            xret_full[firm_ids] = data_month['xret'].values
            xret = weights_on_stocks @ xret_full

            # Hansen-Jagannathan distance
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


def compute_ipca_portfolio_stats(
    ipca_returns: pd.DataFrame,
    ipca_weights: np.ndarray,
    panel: pd.DataFrame,
    panel_id: str,
    model: str,
    K: int,
    chars: list,
    start_month: int,
    end_month: int,
    alpha_lst: list = None,
    include_mkt: bool = False,
    mkt_returns: pd.DataFrame = None,
    burnin: int = None
) -> pd.DataFrame:
    """
    Compute portfolio statistics for IPCA factors.

    Evaluates across alpha grid to find optimal shrinkage for portfolios of IPCA factors.

    Args:
        ipca_returns: IPCA factor returns DataFrame (months x K factors)
        ipca_weights: IPCA factor loadings array (K, N, n_windows)
        panel: Panel data
        panel_id: Panel identifier (e.g., 'bgn_0')
        model: Model name ('bgn', 'kp14', 'gs21')
        K: Number of latent factors
        chars: List of characteristics
        start_month: First month (must be >= burnin + 360 for IPCA)
        end_month: Last month
        alpha_lst: List of ridge penalties (default: [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1])
        include_mkt: Whether to include market factor
        mkt_returns: Market factor returns (if include_mkt=True)
        burnin: Burn-in period (must be from config.BGN_BURNIN/KP14_BURNIN/GS21_BURNIN)

    Returns:
        DataFrame with columns: ['month', 'K', 'alpha', 'include_mkt', 'stdev', 'mean', 'xret', 'sdf_ret', 'hjd']
    """
    if alpha_lst is None:
        alpha_lst = [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1]

    # Load pre-computed SDF moments
    moments, N, moments_start, moments_end = load_precomputed_moments(panel_id)

    # Use moments range if it's more restrictive than provided range
    start_month = max(start_month, moments_start)
    end_month = min(end_month, moments_end)

    results_list = []

    # Get the actual start month of IPCA returns (not from moments file)
    ipca_start = ipca_returns.index.min()

    for month in range(start_month, end_month + 1):
        # Check if month is in IPCA returns
        if month not in ipca_returns.index:
            continue

        # Get pre-computed SDF outputs for this month
        if month not in moments:
            raise KeyError(f"Month {month} not found in pre-computed moments")

        month_moments = moments[month]
        rp = month_moments['rp']
        cond_var = month_moments['cond_var']
        second_moment = month_moments['second_moment']
        second_moment_inv = month_moments['second_moment_inv']
        sdf_ret = month_moments['sdf_ret']

        for alpha in alpha_lst:
            # Compute mean-variance efficient portfolio of IPCA factors
            # Use same approach as DKKM (ridge regression on factor returns)

            # Get past 360 months of IPCA factor returns
            hist_start = month - 360
            hist_end = month - 1

            # Filter available months
            available_months = [m for m in range(hist_start, hist_end + 1)
                               if m in ipca_returns.index]

            if len(available_months) == 0:
                # No history available, skip this month
                continue

            X = ipca_returns.loc[available_months].dropna().to_numpy()

            # Add market if specified
            if include_mkt and mkt_returns is not None:
                mkt_data = mkt_returns.loc[available_months].dropna().to_numpy()
                X = np.column_stack((X, mkt_data))

            y = np.ones(len(X))
            nfeatures = X.shape[1]

            # Compute ridge regression portfolio weights
            if include_mkt:
                # For alpha=0, solve directly
                beta_0 = ridge_regression_grid(X, y, np.array([0]))[:, 0]

                if alpha > 0:
                    # Augment: don't penalize last column (market)
                    X_aug = np.vstack([
                        X,
                        np.sqrt(360 * nfeatures * alpha) * np.eye(X.shape[1])[:-1]
                    ])
                    y_aug = np.concatenate([y, np.zeros(X.shape[1] - 1)])
                    port_of_factors = ridge_regression_grid(X_aug, y_aug, np.array([0]))[:, 0]
                else:
                    port_of_factors = beta_0
            else:
                # Standard ridge regression
                port_of_factors = ridge_regression_grid(
                    X, y, np.array([360 * nfeatures * alpha])
                )[:, 0]

            # Get IPCA loadings for this month
            # ipca_weights shape: (K, N, n_windows)
            # window_idx maps current month to the correct window in ipca_weights
            # IPCA returns/windows start at ipca_start (not start_month!)
            window_idx = month - ipca_start

            if window_idx < 0 or window_idx >= ipca_weights.shape[2]:
                # Skip months outside valid window range
                continue

            # Extract loadings for this window: (K, N) -> transpose to (N, K)
            loadings = ipca_weights[:, :, window_idx].T  # (N, K)

            # Get current month data
            data_month = panel.loc[month].copy()
            firm_ids = data_month.index.to_numpy()

            # Portfolio weights on IPCA factors (use only first K elements if mkt included)
            port_of_ipca_factors = port_of_factors[:K] if include_mkt else port_of_factors

            # Portfolio weights on stocks (partial - only for firms that exist)
            # Extract loadings only for firms that exist at this month
            loadings_partial = loadings[firm_ids, :]  # (n_firms_t, K)
            weights_partial = loadings_partial @ port_of_ipca_factors  # (n_firms_t,)

            # Create full N-dimensional weight vector (for all stocks in SDF)
            weights_on_stocks = np.zeros(N)
            weights_on_stocks[firm_ids] = weights_partial

            # Compute statistics using SDF moments
            stdev = np.sqrt(weights_on_stocks @ cond_var @ weights_on_stocks)
            mean = weights_on_stocks @ rp

            # Realized return: only for available stocks
            xret_full = np.zeros(N)
            xret_full[firm_ids] = data_month['xret'].values
            xret = weights_on_stocks @ xret_full

            # Hansen-Jagannathan distance
            errs = rp - second_moment @ weights_on_stocks
            hjd = np.sqrt(errs @ second_moment_inv @ errs)

            results_list.append({
                'month': month,
                'K': K,
                'alpha': alpha,
                'include_mkt': include_mkt,
                'stdev': stdev,
                'mean': mean,
                'xret': xret,
                'sdf_ret': sdf_ret,
                'hjd': hjd
            })

    return pd.DataFrame(results_list)
