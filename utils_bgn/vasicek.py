__metaclass__ = type

from datetime import datetime
import numpy as np
import pandas as pd
import os
from scipy.optimize import root_scalar, fsolve
from scipy.stats import norm
from scipy.special import roots_laguerre
from config import (
    PI as pi, RBAR as rbar, KAPPA as kappa, SIGMA_R as sigma_r,
    BETA_ZR as beta_zr, SIGMA_Z as sigma_z, CBAR as Cbar
)

# Get path to solution files relative to this module
_SOLFILES_DIR = os.path.join(os.path.dirname(__file__), 'BGN_solfiles')

print(f"started import of vasicek at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")

# parameters for bond pricing, going out max_prds

max_prds = 1500

alpha2 = kappa**np.arange(max_prds+1)
alpha1 = np.concatenate(([0], np.cumsum(alpha2[:-1])))
phi2 = rbar - rbar * kappa**np.arange(max_prds+1)
phi1 = np.concatenate(
    ([0], np.cumsum(phi2[:-1] + 0.5*sigma_z**2 * np.ones(max_prds)))
)
sigma2_sq = (
    sigma_r**2 / (1-kappa**2) 
    - (sigma_r**2 / (1-kappa**2)) * kappa**(2*np.arange(max_prds+1))
)

sigma12 = np.zeros(max_prds+1)
sigma12[0] = 0
for i in range(max_prds):
    sigma12[i+1] = kappa * (sigma12[i] + sigma2_sq[i]) + beta_zr

sigma1_sq = np.concatenate(
    ([0], np.cumsum(sigma2_sq[:-1] + sigma12[:-1] + sigma_z**2 * np.ones(max_prds)))
)

# bond price at maturity k
def B(k, r):
    return np.exp(- alpha1[k]*r - phi1[k] + 0.5*sigma1_sq[k])

# price of consol bond with depreciating coupons, approximating infty = max_prds
def D(r):

    # bond prices at maturities 1, ..., n
    b = np.exp(- alpha1[1:]*r - phi1[1:] + 0.5*sigma1_sq[1:])

    # discounted sum
    return np.sum(pi**np.arange(1, max_prds+1) * b)

# beta distribution parameters

def beta_density(b, beta_star, scale):
    return np.exp((b - beta_star) / scale) / scale

def beta_cdf(b, beta_star, scale):
    return scale * beta_density(b, beta_star, scale)
    
def prob_in_money(r, beta_star, scale):
    return beta_cdf(Cbar + np.log(D(r)), beta_star, scale)
    
def fit_beta_params(x):
    beta_star = x[0]
    scale = np.exp(x[1])
    p1 = prob_in_money(0, beta_star, scale)
    p2 = prob_in_money(rbar, beta_star, scale)
    return np.array([p1-0.1, p2-0.05])

x = fsolve(fit_beta_params, x0 = [1, 1])
beta_star = x[0]
scale = np.exp(x[1])
# print(f"beta_star = {beta_star}")
# print(f"beta_bar = {beta_star - scale}")
# valuing growth options

# interest rate at which option is at the money
def rstar(beta):
    return root_scalar(
        lambda r: Cbar - beta + np.log(D(r)), x0 = rbar
    ).root


def J(r, option_mat, bond_mat, strike):
    num = (
        np.log(B(option_mat+bond_mat, r))
        - np.log(B(option_mat, r))
        - np.log(B(bond_mat, strike))
        + 0.5 * alpha1[bond_mat]**2 * sigma2_sq[option_mat]
    )
    d1 = num / (alpha1[bond_mat]*np.sqrt(sigma2_sq[option_mat]))
    d2 = d1 - alpha1[bond_mat]*np.sqrt(sigma2_sq[option_mat])
    return (
        B(option_mat+bond_mat, r) * norm.cdf(d1) 
        - B(option_mat, r) * B(bond_mat, strike) * norm.cdf(d2)
    )

## replace integralJ function by this code:
nodes, weights = roots_laguerre(100)

rstar_nodes = np.zeros((len(nodes), ))
beta_nodes = beta_star - scale*nodes
for n in range(len(nodes)):
    rstar_nodes[n] = rstar(beta_nodes[n])

def integralJ(r, option_mat, bond_mat):
    return np.sum(weights * J(r, option_mat, bond_mat, rstar_nodes) * np.exp(-beta_nodes))

def inner_sum(r, option_mat):
    integrals = np.array([integralJ(r, option_mat, k) for k in range(1, 401)])
    return np.sum(integrals * pi**np.arange(1, 401))

def Jstar(r):
    return np.sum([inner_sum(r, s) for s in range(1, 951)])

print(f"finished import of vasicek at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")

'''
nsim = int(1.0e8)
r = np.zeros(nsim+1)
r[0] = rbar
xi = norm.rvs(size=nsim)
const = (1-kappa) * rbar
for i in range(nsim):
    r[i+1] = kappa*r[i] + const + sigma_r * xi[i]

r = pd.Series(r)
rmin = r.quantile(1-0.99995)
rmax = r.quantile(0.99995)
print(f"99.99% of the distribution is between {rmin} and {rmax}")

grid1 = np.linspace(rmin, rmax, 51)
J1 = np.array([Jstar(r) for r in grid1])

maxerr = 1
iter = 0
while maxerr > 0.0001:
    print(iter)

    grid2 = 0.5*(grid1[:-1] + grid1[1:])
    J2 = np.array([Jstar(r) for r in grid2])

    grid = np.concatenate((grid1, grid2))
    Jvals = np.concatenate((J1, J2))

    df = pd.DataFrame(
        {"r": grid, "J": Jvals}
    )
    df = df.sort_values(by="r")
    df.to_csv(os.path.join(_SOLFILES_DIR, 'Jstar.csv'), index=False)

    J2hat = np.interp(grid2, grid1, J1)
    err = np.abs(np.array(J2) - J2hat) / J2
    maxerr = np.max(err) 

    grid1 = df.r.to_numpy()
    J1 = df.J.to_numpy()

    iter += 1 


'''