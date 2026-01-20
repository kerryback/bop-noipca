"""
Centralized configuration for the refactored codebase.
All parameters in one place for easy modification.
"""

import numpy as np
import os

# =============================================================================
# PANEL DIMENSIONS - CONFIGURE THESE FIRST
# =============================================================================

N = 1000 # Number of firms (set to 50 for testing)
T = 721   # Number of time periods (excluding burnin) (set to 400 for testing)
BGN_BURNIN = 200  # BGN burnin period
KP14_BURNIN = 200  # KP14 burnin period
GS21_BURNIN = 200  # GS21 burnin period
N_JOBS = 24  # Number of parallel jobs
NUMBA_NUM_THREADS = 1  # Numba threads per job (1 = no nested parallelism)

# =============================================================================
# DATA DIRECTORY CONFIGURATION
# =============================================================================

# Get the directory containing this config file (root/)
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directory for permanent output files (fama, dkkm results)
# Relative to root/ (outputs/ subdirectory within root/)
DATA_DIR = os.path.join(_CONFIG_DIR, 'outputs')

# Temporary directory for intermediate files (panel, moments, chunks)
# These files are deleted based on KEEP_PANEL and KEEP_MOMENTS flags
# Default: same as DATA_DIR (can be overridden by set_temp_dir())
TEMP_DIR = DATA_DIR

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)


def set_temp_dir(temp_path):
    """
    Set the temporary directory path and create it if needed.

    Args:
        temp_path: Path to temporary directory (e.g., '/opt/scratch/keb7')
    """
    global TEMP_DIR
    TEMP_DIR = temp_path
    os.makedirs(TEMP_DIR, exist_ok=True)
    print(f"[CONFIG] TEMP_DIR set to: {TEMP_DIR}")


def set_n_jobs(n_jobs):
    """
    Set the number of parallel jobs.

    Args:
        n_jobs: Number of parallel jobs
    """
    global N_JOBS
    N_JOBS = n_jobs
    print(f"[CONFIG] N_JOBS set to: {N_JOBS}")


def set_jgsrc1_config():
    """
    Configure settings for running on jgsrc1 server.
    Sets TEMP_DIR to /opt/scratch/keb7 and N_JOBS to 10.
    """
    set_temp_dir('/opt/scratch/keb7')
    set_n_jobs(10)

# =============================================================================
# FILE MANAGEMENT FLAGS
# =============================================================================

# Control whether to keep intermediate files after processing
KEEP_PANEL = True    # If False, delete panel files after use
KEEP_MOMENTS = True  # If False, delete moments files after use
KEEP_FACTOR_DETAILS = False  # If False, keep only {method}_stats in factor pickle files

# =============================================================================
# DKKM AND FAMA PARAMETERS
# =============================================================================

# DKKM parameters
INCLUDE_MKT = True  # Include market in DKKM
NMAT = 1  # Number of weight matrices for DKKM
N_DKKM_FEATURES_LIST = [6, 36, 360, 3600, 36000]  # List of feature counts for DKKM computation (removed 360000 due to OOM on 5xlarge)
DKKM_RANK_STANDARDIZE = True  # Rank-standardize characteristics in DKKM

# Shrinkage parameters (Berk-Jameson regularization)
ALPHA_LST_FAMA = [0]  # For Fama methods - OLS ONLY, NO PENALIZATION
ALPHA_LST = [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1]  # For DKKM (BGN/KP)
ALPHA_LST_GS = [0, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]  # For DKKM (GS21)

# Fama-MacBeth preprocessing
STDZ_FM = False  # Standardize characteristics in Fama-MacBeth (False matches original)

# =============================================================================
# MODEL CHARACTERISTICS AND FACTOR NAMES
# =============================================================================

# BGN and KP14 models (5 characteristics)
CHARS_DEFAULT = ["size", "bm", "agr", "roe", "mom"]
FACTOR_NAMES_DEFAULT = ["smb", "hml", "cma", "rmw", "umd"]

# GS21 model (6 characteristics - includes market leverage)
CHARS_GS21 = ["size", "bm", "agr", "roe", "mom", "mkt_lev"]
FACTOR_NAMES_GS21 = ["smb", "hml", "cma", "rmw", "umd", "mkt_lev"]

# =============================================================================
# BGN Model Parameters (from parameters.py)
# =============================================================================
PI = 0.99
RBAR = 0.006236
KAPPA = 0.95
SIGMA_R = 0.002
BETA_ZR = -0.00014
SIGMA_Z = 0.4
CBAR = -3.7
CHAT = np.exp(CBAR)
I = 1
GAMMA_GRID = np.arange(0.5, 1.1, 0.1)

# =============================================================================
# KP14 Model Parameters (from parameters_kp14.py)
# =============================================================================

KP14_DT = 1/12
KP14_MU_X = 0.01
KP14_MU_Z = 0.005
KP14_SIGMA_X = 0.13
KP14_SIGMA_Z = 0.035
KP14_THETA_EPS = 0.35
KP14_SIGMA_EPS = 0.2
KP14_THETA_U = 0.5
KP14_SIGMA_U = 1.5
KP14_DELTA = 0.1
KP14_MU_LAMBDA = 2.0
KP14_SIGMA_LAMBDA = 2.0
KP14_MU_H = 0.075
KP14_MU_L = 0.16
KP14_LAMBDA_H = 2.35
KP14_LAMBDA_L = (1 - KP14_MU_H/(KP14_MU_H + KP14_MU_L)*KP14_LAMBDA_H)/(1 - KP14_MU_H/(KP14_MU_H + KP14_MU_L))
KP14_R = 0.05
KP14_GAMMA_X = 0.69
KP14_GAMMA_Z = -0.35
KP14_ALPHA = 0.85

# Derived KP14 parameters
KP14_PROB_H = KP14_MU_L / (KP14_MU_H + KP14_MU_L)
KP14_CONST = KP14_R + KP14_GAMMA_X * KP14_SIGMA_X + KP14_DELTA - KP14_MU_X
KP14_A_0 = 1 / KP14_CONST
KP14_A_1 = 1 / (KP14_CONST + KP14_THETA_EPS)
KP14_A_2 = 1 / (KP14_CONST + KP14_THETA_U)
KP14_A_3 = 1 / (KP14_CONST + KP14_THETA_EPS + KP14_THETA_U)

KP14_RHO = (KP14_R + KP14_GAMMA_X * KP14_SIGMA_X - KP14_MU_X
            - KP14_ALPHA / (1 - KP14_ALPHA) * (KP14_MU_Z - KP14_GAMMA_Z * KP14_SIGMA_Z - 0.5 * KP14_SIGMA_Z**2)
            - 0.5 * (KP14_ALPHA / (1 - KP14_ALPHA))**2 * KP14_SIGMA_Z**2)
KP14_C = KP14_ALPHA**(1 / (1 - KP14_ALPHA)) * (KP14_ALPHA**(-1) - 1)

# =============================================================================
# GS21 Model Parameters (from parameters_gs21.py)
# =============================================================================
GS21_BETA = 0.994  # subjective discount factor
GS21_PSI = 2  # elasticity of intertemporal substitution
GS21_GAMMA = 10  # risk aversion
GS21_G = 1.14  # size of growth options
GS21_ALPHA = 0.2  # relative size of entrants
GS21_DELTA = 0.02/3  # maintenance investment rate (depreciation)
GS21_RHO_X = 0.95**(1/3)  # AR1 of aggregate shock
GS21_SIGMA_X = 0.012*np.sqrt((1 - 0.95**(3/2))/(1 - 0.95**2))  # vol of aggregate shock
GS21_XBAR = 0  # mean of aggregate shock
GS21_RHO_Z = 0.90**(1/3)  # AR1 of idiosyncratic shock
GS21_SIGMA_Z = 0.16*np.sqrt((1 - 0.9**(3/2))/(1 - 0.9**2))  # vol of idiosyncratic shock
GS21_ZBAR = 0  # mean of idiosyncratic shock
GS21_CHI = 1  # resource cost of default
GS21_TAU = 0.2/3  # effective corporate tax rate
GS21_PHI = 0.4  # bankruptcy cost
GS21_KAPPA_E = 0  # equity issuance cost
GS21_KAPPA_B = 0.004  # bond issuance cost
GS21_ZETA = 0.03/3  # bond refinancing cost
GS21_IMIN = 0  # minimum investment cost
GS21_IMAX = 2000  # maximum investment cost
GS21_R = 0.074830/12  # risk-free rate



# =============================================================================
# MODEL-SPECIFIC MAPPINGS
# =============================================================================

# Characteristics by model
LOADING_KEYS = {
    'bgn': ['A_1_', 'A_2_'],
    'kp14': ['A_1_', 'A_2_'],
    'gs21': ['A_1_']
}

FACTOR_KEYS = {
    'bgn': ['f_1_', 'f_2_'],
    'kp14': ['f_1_', 'f_2_'],
    'gs21': ['f_1_']
}

# Characteristics by model
MODEL_CHARS = {
    'bgn': CHARS_DEFAULT,
    'kp14': CHARS_DEFAULT,
    'gs21': CHARS_GS21
}

# Factor names by model
MODEL_FACTOR_NAMES = {
    'bgn': FACTOR_NAMES_DEFAULT,
    'kp14': FACTOR_NAMES_DEFAULT,
    'gs21': FACTOR_NAMES_GS21
}

# Alpha lists by model
MODEL_ALPHA_LST = {
    'bgn': ALPHA_LST,
    'kp14': ALPHA_LST,
    'gs21': ALPHA_LST_GS
}


def get_model_config(model_name):
    """
    Get configuration dictionary for a specific model.

    Args:
        model_name: Model identifier ('bgn', 'kp14', or 'gs21')

    Returns:
        Dictionary with model configuration
    """
    if model_name not in MODEL_CHARS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Valid models: {list(MODEL_CHARS.keys())}")
        import sys
        sys.exit(1)

    # Get model-specific burnin
    burnin_map = {
        'bgn': BGN_BURNIN,
        'kp14': KP14_BURNIN,
        'gs21': GS21_BURNIN
    }

    return {
        'model': model_name,
        'N': N,
        'T': T,
        'burnin': burnin_map[model_name],
        'chars': MODEL_CHARS[model_name],
        'factor_names': MODEL_FACTOR_NAMES[model_name],
        'n_jobs': N_JOBS,
        'include_mkt': INCLUDE_MKT,
        'nmat': NMAT,
        'n_dkkm_features_list': N_DKKM_FEATURES_LIST,
        'dkkm_rank_standardize': DKKM_RANK_STANDARDIZE,
        'alpha_lst_fama': ALPHA_LST_FAMA,
        'alpha_lst': MODEL_ALPHA_LST[model_name],
        'stdz_fm': STDZ_FM
    }
