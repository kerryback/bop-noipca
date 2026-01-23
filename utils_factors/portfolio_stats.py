"""
Portfolio Statistics - Model Factors

This module contains portfolio statistics computation for model-based factors
(e.g., Taylor/Projection factors from BGN/KP14/GS21 models).

For method-specific portfolio stats:
- Fama: see fama_functions.compute_portfolio_stats()
- DKKM: see dkkm_functions.compute_portfolio_stats()
- IPCA: see ipca_functions.compute_portfolio_stats() [not implemented]
"""

import numpy as np
import pandas as pd
from typing import Dict

from . import fama_functions as fama


def compute_model_portfolio_stats(
    model_premia: Dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    start_month: int,
    end_month: int
) -> pd.DataFrame:
    """
    Compute portfolio statistics for model factors (Taylor/Proj).

    Uses alpha=0 (no penalty) as model factors are already estimated.

    NOTE: This is a placeholder implementation for model-based factors.
    Currently only used for BGN model's theoretical Taylor and Projection factors.

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
