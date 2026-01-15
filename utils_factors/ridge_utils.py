"""
Ridge regression utilities for factor computation.

Consolidated ridge regression functions using eigendecomposition with kernel trick.

Key functions:
- ridge_regression_fast: Single alpha ridge regression
- ridge_regression_grid: Multiple alphas with efficient vectorized computation

Performance:
- For D < T: Standard eigendecomposition of X'X
- For D >= T: Kernel trick using eigendecomposition of XX' (always O(T³))
"""

import numpy as np
import scipy.linalg as linalg


def ridge_regression_fast(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.0
) -> np.ndarray:
    """
    Fast ridge regression using eigenvalue decomposition.

    Handles both T < P and T >= P cases efficiently using kernel trick when needed.

    Args:
        X: (T, P) design matrix
        y: (T,) or (T, 1) response vector
        alpha: Ridge penalty parameter

    Returns:
        beta: (P,) coefficient vector
    """
    y = y.ravel()  # Ensure 1D
    T, P = X.shape

    if T < P:
        # Over-parametrized: use kernel trick
        # beta = X' (XX' + alpha*I)^{-1} y
        try:
            U, d, VT = linalg.svd(X @ X.T, full_matrices=False)
            V = VT.T
            W = X.T @ V @ np.diag(1 / np.sqrt(d))

            # Compute: W @ (d + alpha)^{-1} @ W' @ X' @ y
            XTy = X.T @ y
            WTXTy = W.T @ XTy
            beta = W @ (WTXTy / (d + alpha))

        except linalg.LinAlgError:
            # Fallback to standard approach
            V, d, VT = linalg.svd(X.T @ X, full_matrices=False)
            beta = V @ np.diag(1 / (d + alpha)) @ VT @ X.T @ y

    else:
        # Standard case: beta = (X'X + alpha*I)^{-1} X' y
        V, d, VT = linalg.svd(X.T @ X, full_matrices=False)
        beta = V @ np.diag(1 / (d + alpha)) @ VT @ X.T @ y

    return beta


def ridge_regression_grid(
    signals: np.ndarray,
    labels: np.ndarray,
    shrinkage_list: np.ndarray
) -> np.ndarray:
    """
    Ridge regression for a grid of shrinkage parameters.

    Efficient vectorized computation across multiple penalties using eigendecomposition.
    Automatically uses kernel trick when P > T for computational efficiency.

    Args:
        signals: (T, P) design matrix
        labels: (T,) response vector
        shrinkage_list: (n_alpha,) array of ridge parameters

    Returns:
        betas: (P, n_alpha) coefficient matrix
    """
    t_, p_ = signals.shape
    labels = labels.reshape(-1, 1)

    if p_ < t_:
        # Standard regime: eigendecomposition of X'X
        eigenvalues, eigenvectors = np.linalg.eigh(signals.T @ signals)
        means = signals.T @ labels

        multiplied = eigenvectors.T @ means

        # Compute for all shrinkage values at once
        intermed = np.concatenate([
            (1 / (eigenvalues.reshape(-1, 1) + 360 * z)) * multiplied
            for z in shrinkage_list
        ], axis=1)

        betas = eigenvectors @ intermed

    else:
        # Over-parametrized regime: use kernel trick (XX' instead of X'X)
        eigenvalues, eigenvectors = np.linalg.eigh(signals @ signals.T)
        means = labels

        multiplied = eigenvectors.T @ means

        intermed = np.concatenate([
            (1 / (eigenvalues.reshape(-1, 1) + 360 * z)) * multiplied
            for z in shrinkage_list
        ], axis=1)

        tmp = eigenvectors.T @ signals
        betas = tmp.T @ intermed

    return betas
