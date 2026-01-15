import numpy as np
import pandas as pd
import os
from config import (
    GS21_BETA as beta, GS21_PSI as psi, GS21_GAMMA as gamma,
    GS21_G as g, GS21_ALPHA as alpha, GS21_DELTA as delta,
    GS21_RHO_X as rho_x, GS21_SIGMA_X as sigma_x, GS21_XBAR as xbar,
    GS21_RHO_Z as rho_z, GS21_SIGMA_Z as sigma_z, GS21_ZBAR as zbar,
    GS21_CHI as chi, GS21_TAU as tau, GS21_PHI as phi,
    GS21_KAPPA_E as kappa_e, GS21_KAPPA_B as kappa_b, GS21_ZETA as zeta,
    GS21_IMIN as imin, GS21_IMAX as imax, GS21_R as r,
    GS21_BURNIN as burnin
)
from .sdf_compute_gs21 import *

# Get path to solution files relative to this module
_SOLFILES_DIR = os.path.join(os.path.dirname(__file__), 'GS21_solfiles')

P_up_mat = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'P_up.csv'), header=None)).reshape(zpts, xpts, bpts)
P_down_mat = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'P_down.csv'), header=None)).reshape(zpts, xpts, bpts)

## function to compute derivatives of D and Jstar

P_up_diff = np.zeros_like(P_up_mat)
P_up_diff[:, 1:-1, :] = ((P_up_mat[:, 2:, :] - P_up_mat[:, :-2,:]) / (xgrid[2:] - xgrid[:-2]).reshape((1, xpts-2, 1)))
P_up_diff[:,0,:] = ((P_up_mat[:, 1, :] - P_up_mat[:, 0,:]) / (xgrid[1] - xgrid[0]))
P_up_diff[:,-1,:] = ((P_up_mat[:, -2, :] - P_up_mat[:, -1,:]) /(xgrid[-2] - xgrid[-1]))

P_down_diff = np.zeros_like(P_down_mat)
P_down_diff[:, 1:-1, :] = ((P_down_diff[:, 2:, :] - P_down_diff[:, :-2,:]) / (xgrid[2:] - xgrid[:-2]).reshape((1, xpts-2, 1)))
P_down_diff[:,0,:] = ((P_down_diff[:, 1, :] - P_down_diff[:, 0,:]) / (xgrid[1] - xgrid[0]))
P_down_diff[:,-1,:] = ((P_down_diff[:, -2, :] - P_down_diff[:, -1,:]) /(xgrid[-2] - xgrid[-1]))

P_up_diff = RegularGridInterpolator(points=(zgrid, xgrid, bgrid), values=P_up_diff,method='linear',bounds_error=False,  fill_value=None)
P_down_diff = RegularGridInterpolator(points=(zgrid, xgrid, bgrid), values=P_down_diff,method='linear',bounds_error=False,  fill_value=None)

# function to compute loadings based on Taylor
def loadings_Taylor(k, b, x, z, cutoff, eta, P, P_ex, erets, rets, op_cashflow, default, N):
    Ex = np.repeat(rho_x*x.reshape(-1, 1), N, axis = 1)
    x = np.repeat(x.reshape(-1, 1), N, axis = 1)
    Ez = rho_z*z

    A_x_I_no = g*(cutoff - imin)/(imax - imin)*(eta == 0)*((1 - zeta)*P_down_diff((Ez, Ex,  b/g)) + zeta*P_up_diff((Ez, Ex,  b/g)))
    A_x_0_no = (imax - cutoff)/(imax - imin)*(eta == 0)*((1 - zeta)*P_down_diff((Ez, Ex,  b)) + zeta*P_up_diff((Ez, Ex,  b)))
    A_x_I_ref = g*(cutoff - imin)/(imax - imin)*(eta == 1)*((1 - zeta)*P_down_diff((Ez, Ex,  b_refin_I((z, x, b)))) + zeta*P_up_diff((Ez, Ex, b_refin_I((z, x, b)))))
    A_x_0_ref = (imax - cutoff)/(imax - imin)*(eta == 1)*((1 - zeta)*P_down_diff((Ez, Ex,  b_refin_0((z, x, b)))) + zeta*P_up_diff((Ez, Ex,  b_refin_0((z, x, b)))))
     
    A_x = A_x_I_no + A_x_0_no + A_x_I_ref + A_x_0_ref
    return A_x/P_ex

# function to compute loadings based on projection
def loadings_projection(k, b, x, z, cutoff, eta, P, P_ex, erets, rets, op_cashflow, default, N, T):

    # function to compute expected product of (capital gain of) return and x'
    def integ(icutoff, eta, bi, zi, x, n = 10):
        # GH nodes
        nodes, w = hermgauss(n) 
        nodes = np.sqrt(2.0) * nodes

        # reshape inputs
        bi = bi.reshape(-1, 1)
        zi = zi.reshape(-1, 1)
        eta = eta.reshape(-1, 1)
        icutoff = icutoff.reshape(-1, 1)
        x = x*np.ones(zi.shape)
        
        # for each zi, and for x compute n^2-grid of next periods values
        zi_new = rho_z*zi + sigma_z*np.repeat(nodes, n).reshape(-1, 1).T
        x_new = rho_x*x + sigma_x*np.tile(nodes, n).reshape(-1, 1).T
        
        # compute corresopnding weights
        w_zi = np.repeat(w, n).reshape(-1, 1).T
        w_x = np.tile(w, n).reshape(-1, 1).T
        w = w_zi*w_x

        # compute term for expected next periods prices
        bi = np.repeat(bi, n**2, axis = 1)
        Pup_bi = (g*(icutoff - imin)/(imax - imin)*(eta == 0)*P_up((zi_new, x_new,  bi/g)) +                        # invest and no refinance
                    (imax - icutoff)/(imax - imin)*(eta == 0)*P_up(( zi_new, x_new, bi)) +                          # no invest and no refinance
                    g*(icutoff - imin)/(imax - imin)*(eta == 1)*P_up(( zi_new, x_new, b_refin_I((zi, x, bi)))) +    # invest and refinance
                    (imax - icutoff)/(imax - imin)*(eta == 1)*P_up(( zi_new, x_new, b_refin_0((zi, x, bi)))))       # no invest and refinance
        Pdn_bi = (g*(icutoff - imin)/(imax - imin)*(eta == 0)*P_down((zi_new, x_new,  bi/g)) +                      # invest and no refinance
                    (imax - icutoff)/(imax - imin)*(eta == 0)*P_down((zi_new, x_new,  bi)) +                        # no invest and no refinance
                    g*(icutoff - imin)/(imax - imin)*(eta == 1)*P_down((zi_new, x_new,  b_refin_I((zi, x, bi)))) +  # invest and refinance
                    (imax - icutoff)/(imax - imin)*(eta == 1)*P_down(( zi_new, x_new, b_refin_0((zi, x, bi)))))     # no invest and refinance

        # integrate - first over eta, then over z' and x'
        refin =      zeta        * Pup_bi 
        no_refin   = (1-zeta)    * Pdn_bi
        exp_Px = (w * x_new * (refin + no_refin)).sum(axis=1) / np.pi # (normalize by pi since it's a double integral)
        exp_P = (w * (refin + no_refin)).sum(axis=1) / np.pi # (normalize by pi since it's a double integral)

        return  exp_Px, exp_P
    
    A_x = np.zeros((T+1, N))
    for t in range(T+1):
        res1, res2 = integ(cutoff[t], eta[t], b[t], z[t], x[t], n = 10)
        A_x[t] = (res1 - res2*(rho_x*x[t]))/P_ex[t]
    
    return A_x/sigma_x**2
    
