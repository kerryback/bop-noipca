import numpy as np
import pandas as pd
import os
from scipy import interpolate
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

from numpy.polynomial.hermite import hermgauss
import scipy.special
from scipy.sparse import csr_matrix, diags, kron, vstack
from scipy.interpolate import RegularGridInterpolator

# define functions outputted from matlab

# Get path to solution files relative to this module
_SOLFILES_DIR = os.path.join(os.path.dirname(__file__), 'GS21_solfiles')

# read in grids
zgrid = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'zgrid.csv'), header=None)).reshape(-1)
xgrid = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'xgrid.csv'), header=None)).reshape(-1)
igrid = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'igrid.csv'), header=None)).reshape(-1)
bgrid = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'bgrid.csv'), header=None)).reshape(-1)
zpts = len(zgrid)
xpts = len(xgrid)
ipts = len(igrid)
bpts = len(bgrid)

# read in function values and reshape
z_cut_up = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'z_cut_up.csv'), header=None)).reshape(xpts, bpts)
z_cut_down = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'z_cut_down.csv'), header=None)).reshape(xpts, bpts)
i_cut_up = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'i_cut_up.csv'), header=None)).reshape(zpts, xpts, bpts)
i_cut_down = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'i_cut_down.csv'), header=None)).reshape(zpts, xpts, bpts)
Q_I = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'Q_I.csv'), header=None)).reshape(zpts, xpts, bpts)
Q_0 = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'Q_0.csv'), header=None)).reshape(zpts, xpts, bpts)
P_up = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'P_up.csv'), header=None)).reshape(zpts, xpts, bpts)
P_down = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'P_down.csv'), header=None)).reshape(zpts, xpts, bpts)
PI_up = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'PI_up.csv'), header=None)).reshape(zpts, xpts, bpts)
PI_down = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'PI_down.csv'), header=None)).reshape(zpts, xpts, bpts)
P0_up = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'P0_up.csv'), header=None)).reshape(zpts, xpts, bpts)
P0_down = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'P0_down.csv'), header=None)).reshape(zpts, xpts, bpts)
b_refin_0 = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'b_refin_0.csv'), header=None)).reshape(zpts, xpts, bpts)
b_refin_I = np.array(pd.read_csv(os.path.join(_SOLFILES_DIR, 'b_refin_I.csv'), header=None)).reshape(zpts, xpts, bpts)


def pos_interp(interp):
    """Wrap an interpolator so that its output is always >= 0."""
    def wrapped(x):
        val = interp(x)
        return np.maximum(val, 0.0)
    return wrapped

# Define interpolators for functions
z_cut_up = RegularGridInterpolator(points=(xgrid, bgrid), values=z_cut_up,method='linear',bounds_error=False,  fill_value=None)
z_cut_down = RegularGridInterpolator(points=(xgrid, bgrid), values=z_cut_down,method='linear',bounds_error=False,  fill_value=None)
i_cut_up = pos_interp(RegularGridInterpolator(points=(zgrid, xgrid, bgrid), values=i_cut_up,method='linear',bounds_error=False,  fill_value=None))
i_cut_down = pos_interp(RegularGridInterpolator(points=(zgrid, xgrid, bgrid), values=i_cut_down,method='linear',bounds_error=False,  fill_value=None))
Q_I = pos_interp(RegularGridInterpolator(points=(zgrid, xgrid, bgrid), values=Q_I,method='linear',bounds_error=False,  fill_value=None))
Q_0 = pos_interp(RegularGridInterpolator(points=(zgrid, xgrid, bgrid), values=Q_0,method='linear',bounds_error=False,  fill_value=None))
P_up = pos_interp(RegularGridInterpolator(points=(zgrid, xgrid, bgrid), values=P_up,method='linear',bounds_error=False,  fill_value=None))
P_down = pos_interp(RegularGridInterpolator(points=(zgrid, xgrid, bgrid), values=P_down,method='linear',bounds_error=False,  fill_value=None))
PI_up = pos_interp(RegularGridInterpolator(points=(zgrid, xgrid, bgrid), values=PI_up,method='nearest',bounds_error=False,  fill_value=None))
PI_down = pos_interp(RegularGridInterpolator(points=(zgrid, xgrid, bgrid), values=PI_down,method='linear',bounds_error=False,  fill_value=None))
P0_up = pos_interp(RegularGridInterpolator(points=(zgrid, xgrid, bgrid), values=P0_up,method='linear',bounds_error=False,  fill_value=None))
P0_down = pos_interp(RegularGridInterpolator(points=(zgrid, xgrid, bgrid), values=P0_down,method='linear',bounds_error=False,  fill_value=None))
b_refin_0 = pos_interp(RegularGridInterpolator(points=(zgrid, xgrid, bgrid), values=b_refin_0,method='linear',bounds_error=False,  fill_value=None))
b_refin_I = pos_interp(RegularGridInterpolator(points=(zgrid, xgrid, bgrid), values=b_refin_I,method='linear',bounds_error=False,  fill_value=None))

# goal is to compute NxN matrix of E[Ri Rj]
# function to integrate over zi, zj, x using GH quadrature with n points
def integ_zi_zj_x(N, Pi, i_cutoff, bi, zi, eta, x, n = 10):
    # compute GH weights and nodes
    nodes, w = hermgauss(n) 
    nodes = np.sqrt(2.0) * nodes

    # reshape inputs
    bi_prev = bi.reshape(-1, 1)
    eta = eta.reshape(-1, 1)
    zi_prev = zi.reshape(-1, 1)
    i_cutoff = i_cutoff.reshape(-1, 1)
    x_prev = x*np.ones(zi_prev.shape)

    # gives N by n matrix. Each row has n^2 values of z', x' given today's z, x
    zi = rho_z*zi_prev + sigma_z*np.tile(nodes, n).reshape(-1, 1).T
    x = rho_x*x_prev + sigma_x*np.repeat(nodes, n).reshape(-1, 1).T
    
    # compute corresponding weights
    w_z = np.tile(w, n).reshape(-1, 1).T
    w_x = w.reshape(-1, 1).T
    w = w_z * np.repeat(w, n).reshape(-1, 1).T

    # integrate todays cash flows over i
    prob_i  = (i_cutoff - imin)/(imax - imin)
    
    mean_i_cost = (i_cutoff + imin)/2
    '''
    prof_I_down = (((1 - tau)*np.exp(x_prev + zi_prev) - delta)-mean_i_cost - (1 - tau)*bi_prev)                                        # invest and eta = 0
    prof_0_down = (((1 - tau)*np.exp(x_prev + zi_prev) - delta) - (1 - tau)*bi_prev)                                                    # no invest and eta = 0
    prof_I_up = (((1 - tau)*np.exp(x_prev + zi_prev) - delta)-mean_i_cost - (1 - tau)*bi_prev +                                         # invest and eta = 1            
                  (1 - kappa_b)*Q_I((zi_prev, x_prev, b_refin_I((zi_prev, x_prev, bi_prev)))) - Q_I((zi_prev, x_prev, bi_prev/g)))      
    prof_0_up = (((1 - tau)*np.exp(x_prev + zi_prev) - delta) - (1 - tau)*bi_prev +                                                     # no invest and eta = 1
                 (1 - kappa_b)*Q_0((zi_prev, x_prev, b_refin_0((zi_prev, x_prev, bi_prev)))) - Q_0((zi_prev, x_prev, bi_prev)))
    
    # aggregate today's cash flow across today's eta
    cashflow_i = (prob_i*((eta == 0)*prof_I_down*np.exp(r) + (eta == 1)*prof_I_up*np.exp(r))+
                (1 - prob_i)*((eta == 0)*prof_0_down*np.exp(r)+ (eta == 1)*prof_0_up*np.exp(r))) 
    '''
    # reshape debt values
    bi_re = np.repeat(bi_prev, n**2, axis = 1)
    bi_0_ref_re = np.repeat(b_refin_0((zi_prev, x_prev, bi_prev)), n**2, axis = 1)
    bi_I_ref_re = np.repeat(b_refin_I((zi_prev, x_prev, bi_prev)), n**2, axis = 1)

    # compute expected future values, integrating over i
    Pup = (prob_i*((eta == 0)*g*P_up((zi, x, bi_re/g))                           # invest and eta = 0, eta' = 1    
                  + (eta == 1)*g*P_up((zi, x ,bi_I_ref_re))) +                   # invest and eta = 1, eta' = 1           
                      (1 - prob_i)*((eta == 0)*P_up((zi, x, bi_re))              # no invest and eta = 0, eta' = 1
                                    + (eta == 1)*P_up(( zi, x, bi_0_ref_re))))   # no invest and eta = 1, eta' = 1
    Pdown = (prob_i*((eta == 0)*g*P_down((zi, x, bi_re/g))                       # invest and eta = 0, eta' = 0
                  + (eta == 1)*g*P_down((zi, x,bi_I_ref_re ))) +                 # invest and eta = 1, eta' = 0
                      (1 - prob_i)*((eta == 0)*P_down((zi, x, bi_re))            # no invest and eta = 0, eta' = 0
                                    + (eta == 1)*P_down((zi, x, bi_0_ref_re))))  # no invest and eta = 1, eta' = 0

    # aggregate over eta'
    # payoff is N by n^2
    payoff =  zeta*Pup + (1 - zeta)*Pdown #cashflow_i +

    # integrate over z
    weighted = payoff*w_z
    weighted = weighted.reshape(N, n, n, order='F').sum(axis=1) #integrate over z
    
    # weighted is now N by n. Rows are firms, columns are realizations of x'.

    # integrate over x
    exp_weighted = np.tile(weighted, (N, 1))*np.repeat(weighted, N, axis = 0)*w_x

    ### separate case for i = j

    '''
    # recompute profits, but without investment costs, which are adjusted for later
    prof_I_down = prof_I_down + mean_i_cost
    prof_0_down = prof_0_down
    prof_I_up = prof_I_up + mean_i_cost
    prof_0_up = prof_0_up

    # separate by eta', when no investment
    P_up_0 = (eta == 0)*P_up((zi, x, bi_re)) + (eta == 1)*P_up((zi, x, bi_0_ref_re))
    P_down_0 = (eta == 0)*P_down((zi, x, bi_re)) + (eta == 1)*P_down((zi, x, bi_0_ref_re))

    # compute expected cashflow + P' from no investment
    P_0_sq = ((1-prob_i)*(((eta == 1)*prof_0_up*np.exp(r) + (eta == 0)*prof_0_down*np.exp(r) + P_up_0)**2*zeta) + 
              (1-prob_i)*(((eta == 1)*prof_0_up*np.exp(r) + (eta == 0)*prof_0_down*np.exp(r) + P_down_0)**2*(1 - zeta)))

    # separate by eta', with investment 
    P_up_I = ((eta == 0)*g*P_up((zi, x, bi_re/g)) + (eta == 1)*g*P_up((zi, x, bi_I_ref_re)))
    P_down_I = ((eta == 0)*g*P_down((zi, x, bi_re/g)) + (eta == 1)*g*P_down((zi, x, bi_I_ref_re)))

    # compute expected cashflow + P' from investment
    P_I_sq = prob_i*(((eta == 1)*prof_I_up*np.exp(r) + (eta == 0)*prof_I_down*np.exp(r) + P_up_I)**2*zeta) + \
        prob_i*(((eta == 1)*prof_I_up*np.exp(r) + (eta == 0)*prof_I_down*np.exp(r) + P_down_I)**2*(1 - zeta)) -\
            2*prob_i*(((eta == 1)*prof_I_up*np.exp(r) + (eta == 0)*prof_I_down*np.exp(r) + P_up_I)*zeta +\
                       ((eta == 1)*prof_I_up*np.exp(r) + (eta == 0)*prof_I_down*np.exp(r) + P_down_I)*(1 - zeta))*mean_i_cost +\
            prob_i*(imin**2 + imin*i_cutoff + i_cutoff**2)/3
    '''

    # separate by eta', when no investment
    P_up_0 = (eta == 0)*P_up((zi, x, bi_re)) + (eta == 1)*P_up((zi, x, bi_0_ref_re))
    P_down_0 = (eta == 0)*P_down((zi, x, bi_re)) + (eta == 1)*P_down((zi, x, bi_0_ref_re))

    # compute expected cashflow + P' from no investment
    P_0_sq = (1-prob_i)*(P_up_0**2*zeta + P_down_0**2*(1 - zeta))

    # separate by eta', with investment 
    P_up_I = ((eta == 0)*g*P_up((zi, x, bi_re/g)) + (eta == 1)*g*P_up((zi, x, bi_I_ref_re)))
    P_down_I = ((eta == 0)*g*P_down((zi, x, bi_re/g)) + (eta == 1)*g*P_down((zi, x, bi_I_ref_re)))

    # compute expected cashflow + P' from investment
    P_I_sq = prob_i*(P_up_I**2*zeta + P_down_I**2*(1 - zeta))
    # compute final expectation
    
    ERR = exp_weighted.sum(axis = 1).reshape((N, N))/np.outer(Pi, Pi)/np.pi**(3/2)          # normalize by pi^3.5 since it's a triple integral    
    ERR[np.arange(N),np.arange(N)]  = (((P_0_sq + P_I_sq)*w).sum(axis = 1)/Pi**2)/np.pi     # normalize by pi since it's a double integral

    return ERR

def sdf_compute(N, T, arr_tuple):
    k, b, x, z, cutoff, eta, P, P_ex, eret, ret, op_cashflow, default, loadings_x_taylor, loadings_x_proj = arr_tuple
    def sdf_loop(t, iter):
        t = t+1
        ER = np.zeros((N+1, N+1))
        ER[1:, 1:] = integ_zi_zj_x(N, P_ex[t], cutoff[t], b[t], z[t], eta[t], x[t], n = 10)
        # store entries when at least one of assets in risk free
        ER[0, 0] = np.exp(2 * r)
        ER[0, 1:] = (eret[t, :] + 1) * np.exp(r)
        ER[1:, 0] = (eret[t, :] + 1) * np.exp(r)

        # SDF portfolio and returns
        
        # invert ER
        try:
            # First attempt to solve
            port = scipy.linalg.solve(ER, np.ones((N + 1, 1)), assume_a="pos").reshape(-1) 
        except Exception as e:
            print(f"An error occurred: {e}. Perturbing ER.")
            # Perturb ER and retry
            ER += np.eye(ER.shape[0]) * 1e-6
            try:
                # Second attempt after perturbation
                port = scipy.linalg.solve(ER, np.ones((N + 1, 1))).reshape(-1)
            except Exception as e:
                # Handle failure after retry
                print(f"Second attempt failed: {e}. Using fallback value for port.")
                port = np.full((N+1, ), np.nan)

        #import pdb; pdb.set_trace()
        port /= port.sum()
        sdf_ret = -(port[1:] * (1 + ret[t, :] - np.exp(r))).sum()

        # conditional variance of risky assets
        cond_var = ER[1:, 1:] - np.outer(1 + eret[t, :], 1 + eret[t, :])  
        max_sr = -(port[1:] * (1 + eret[t, :] - np.exp(r))).sum() / np.sqrt(
            port[1:] @ (cond_var @ port[1:])
        )

        return (
            sdf_ret,
            max_sr,
            1 + eret[t, :] - np.exp(r),
            cond_var,
        )

    return sdf_loop
