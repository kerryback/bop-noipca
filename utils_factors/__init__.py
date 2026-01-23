"""Factor computation utilities.

Shared utilities for factor computation scripts (run_fama.py, run_dkkm.py, etc.).

Modules:
    factor_utils: Shared utility functions for loading, preparing, and saving data
    factor_utils_numba: Numba-accelerated versions of factor_utils functions
    ridge_utils: Ridge regression utilities using eigendecomposition with kernel trick
    sdf_utils: SDF conditional moments loading utilities
    fama_functions: Fama-French and Fama-MacBeth factor computation and portfolio stats
    dkkm_functions: DKKM (Random Fourier Features) factor computation and portfolio stats
    dkkm_functions_numba: Numba-accelerated DKKM functions
    portfolio_stats: Model-based factor portfolio statistics (Taylor/Projection)
    ipca_functions: IPCA factor computation and portfolio stats (not implemented)
"""

from . import factor_utils
from . import factor_utils_numba
from . import ridge_utils
from . import sdf_utils
from . import fama_functions
from . import dkkm_functions
from . import dkkm_functions_numba
from . import portfolio_stats
from . import ipca_functions

__all__ = [
    'factor_utils',
    'factor_utils_numba',
    'ridge_utils',
    'sdf_utils',
    'fama_functions',
    'dkkm_functions',
    'dkkm_functions_numba',
    'portfolio_stats',
    'ipca_functions',
]
