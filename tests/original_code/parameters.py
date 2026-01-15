import numpy as np
import sys
from pathlib import Path

# Import burnin from main config.py (two directories up from tests/original_code/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BGN_BURNIN as burnin

pi = 0.99
rbar = 0.006236
kappa = 0.95
sigma_r = 0.002
beta_zr = - 0.00014
sigma_z = 0.4
Cbar = -3.7
Chat = np.exp(Cbar)
I = 1
gamma_grid = np.arange(0.5, 1.1, 0.1)
chars = ["size", "bm", "agr", "roe", "mom"]
names = ["smb", "hml", "cma", "rmw", "umd"]
nchars = len(chars)