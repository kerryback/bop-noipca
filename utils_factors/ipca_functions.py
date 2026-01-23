"""
IPCA (Instrumented Principal Component Analysis) functions.

NOTE: IPCA is not currently implemented in this workflow.
This file is a placeholder for future IPCA implementation.
"""

import numpy as np
import pandas as pd
from .ridge_utils import ridge_regression_grid
from .sdf_utils import load_precomputed_moments


# IPCA CODE COMMENTED OUT - NOT RUNNING IPCA
# def compute_portfolio_stats(
#     ipca_returns: pd.DataFrame,
#     ipca_weights: np.ndarray,
#     panel: pd.DataFrame,
#     panel_id: str,
#     model: str,
#     K: int,
#     chars: list,
#     start_month: int,
#     end_month: int,
#     alpha_lst: list = None,
#     include_mkt: bool = False,
#     mkt_returns: pd.DataFrame = None,
#     burnin: int = None
# ) -> pd.DataFrame:
#     """
#     Compute portfolio statistics for IPCA factors.
#
#     Evaluates across alpha grid to find optimal shrinkage for portfolios of IPCA factors.
#
#     Args:
#         ipca_returns: IPCA factor returns DataFrame (months x K factors)
#         ipca_weights: IPCA factor loadings array (K, N, n_windows)
#         panel: Panel data
#         panel_id: Panel identifier (e.g., 'bgn_0')
#         model: Model name ('bgn', 'kp14', 'gs21')
#         K: Number of latent factors
#         chars: List of characteristics
#         start_month: First month (must be >= burnin + 360 for IPCA)
#         end_month: Last month
#         alpha_lst: List of ridge penalties (default: [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1])
#         include_mkt: Whether to include market factor
#         mkt_returns: Market factor returns (if include_mkt=True)
#         burnin: Burn-in period (must be from config.BGN_BURNIN/KP14_BURNIN/GS21_BURNIN)
#
#     Returns:
#         DataFrame with columns: ['month', 'K', 'alpha', 'include_mkt', 'stdev', 'mean', 'xret', 'sdf_ret', 'hjd']
#     """
#     if alpha_lst is None:
#         alpha_lst = [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1]
#
#     # Load pre-computed SDF moments
#     moments, N, moments_start, moments_end = load_precomputed_moments(panel_id)
#
#     # Use moments range if it's more restrictive than provided range
#     start_month = max(start_month, moments_start)
#     end_month = min(end_month, moments_end)
#
#     results_list = []
#
#     # Get the actual start month of IPCA returns (not from moments file)
#     ipca_start = ipca_returns.index.min()
#
#     for month in range(start_month, end_month + 1):
#         # Check if month is in IPCA returns
#         if month not in ipca_returns.index:
#             continue
#
#         # Get pre-computed SDF outputs for this month
#         if month not in moments:
#             raise KeyError(f"Month {month} not found in pre-computed moments")
#
#         month_moments = moments[month]
#         rp = month_moments['rp']
#         cond_var = month_moments['cond_var']
#         second_moment = month_moments['second_moment']
#         second_moment_inv = month_moments['second_moment_inv']
#         sdf_ret = month_moments['sdf_ret']
#
#         for alpha in alpha_lst:
#             # Compute mean-variance efficient portfolio of IPCA factors
#             # Use same approach as DKKM (ridge regression on factor returns)
#
#             # Get past 360 months of IPCA factor returns
#             hist_start = month - 360
#             hist_end = month - 1
#
#             # Filter available months
#             available_months = [m for m in range(hist_start, hist_end + 1)
#                                if m in ipca_returns.index]
#
#             if len(available_months) == 0:
#                 # No history available, skip this month
#                 continue
#
#             X = ipca_returns.loc[available_months].dropna().to_numpy()
#
#             # Add market if specified
#             if include_mkt and mkt_returns is not None:
#                 mkt_data = mkt_returns.loc[available_months].dropna().to_numpy()
#                 X = np.column_stack((X, mkt_data))
#
#             y = np.ones(len(X))
#             nfeatures = X.shape[1]
#
#             # Compute ridge regression portfolio weights
#             if include_mkt:
#                 # For alpha=0, solve directly
#                 beta_0 = ridge_regression_grid(X, y, np.array([0]))[:, 0]
#
#                 if alpha > 0:
#                     # Augment: don't penalize last column (market)
#                     X_aug = np.vstack([
#                         X,
#                         np.sqrt(360 * nfeatures * alpha) * np.eye(X.shape[1])[:-1]
#                     ])
#                     y_aug = np.concatenate([y, np.zeros(X.shape[1] - 1)])
#                     port_of_factors = ridge_regression_grid(X_aug, y_aug, np.array([0]))[:, 0]
#                 else:
#                     port_of_factors = beta_0
#             else:
#                 # Standard ridge regression
#                 port_of_factors = ridge_regression_grid(
#                     X, y, np.array([360 * nfeatures * alpha])
#                 )[:, 0]
#
#             # Get IPCA loadings for this month
#             # ipca_weights shape: (K, N, n_windows)
#             # window_idx maps current month to the correct window in ipca_weights
#             # IPCA returns/windows start at ipca_start (not start_month!)
#             window_idx = month - ipca_start
#
#             if window_idx < 0 or window_idx >= ipca_weights.shape[2]:
#                 # Skip months outside valid window range
#                 continue
#
#             # Extract loadings for this window: (K, N) -> transpose to (N, K)
#             loadings = ipca_weights[:, :, window_idx].T  # (N, K)
#
#             # Get current month data
#             data_month = panel.loc[month].copy()
#             firm_ids = data_month.index.to_numpy()
#
#             # Portfolio weights on IPCA factors (use only first K elements if mkt included)
#             port_of_ipca_factors = port_of_factors[:K] if include_mkt else port_of_factors
#
#             # Portfolio weights on stocks (partial - only for firms that exist)
#             # Extract loadings only for firms that exist at this month
#             loadings_partial = loadings[firm_ids, :]  # (n_firms_t, K)
#             weights_partial = loadings_partial @ port_of_ipca_factors  # (n_firms_t,)
#
#             # Create full N-dimensional weight vector (for all stocks in SDF)
#             weights_on_stocks = np.zeros(N)
#             weights_on_stocks[firm_ids] = weights_partial
#
#             # Compute statistics using SDF moments
#             stdev = np.sqrt(weights_on_stocks @ cond_var @ weights_on_stocks)
#             mean = weights_on_stocks @ rp
#
#             # Realized return: only for available stocks
#             xret_full = np.zeros(N)
#             xret_full[firm_ids] = data_month['xret'].values
#             xret = weights_on_stocks @ xret_full
#
#             # Hansen-Jagannathan distance
#             errs = rp - second_moment @ weights_on_stocks
#             hjd = np.sqrt(errs @ second_moment_inv @ errs)
#
#             results_list.append({
#                 'month': month,
#                 'K': K,
#                 'alpha': alpha,
#                 'include_mkt': include_mkt,
#                 'stdev': stdev,
#                 'mean': mean,
#                 'xret': xret,
#                 'sdf_ret': sdf_ret,
#                 'hjd': hjd
#             })
#
#     return pd.DataFrame(results_list)
