import numpy as np
import sys
from pathlib import Path

# Import burnin from main config.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import GS21_BURNIN as burnin

beta = 0.994 # subjective discount factor
psi = 2 # elasticity of intertemp subst
gamma = 10 # risk aversion
g = 1.14 # size of growth options
alpha = 0.2 # relative size of entrants
delta = 0.02/3 # maintenance investment rate (depreciation)
rho_x = 0.95**(1/3) # AR1 of aggregate shock
sigma_x = 0.012*np.sqrt((1 - 0.95**(3/2))/(1 - 0.95**2)) # vol of aggregate shock
xbar = 0 # mean of aggregate shock
rho_z = 0.90**(1/3) # AR1 of idio shock
sigma_z = 0.16*np.sqrt((1 - 0.9**(3/2))/(1 - 0.9**2)) # vol of idio shock
zbar = 0 # mean of idio shock
chi = 1 # resource cost of default
tau = 0.2/3 # effective corporate tax rate
phi = 0.4 # bankruptcy cost
kappa_e = 0 #0.025 # equity issuance cost
kappa_b = 0.004 # bond issuance cost
zeta = 0.03/3 # bond refinancing cost
imin = 0 # minimum investment cost
imax = 2000 # maximum investment cost
r = 0.074830/12 # risk-free rate
