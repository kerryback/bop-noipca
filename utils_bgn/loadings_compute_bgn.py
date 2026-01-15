import numpy as np
import pandas as pd
import scipy.linalg as linalg

from . import panel_functions_bgn as bgn
from .sdf_compute_bgn import *
from config import (
    PI as pi, RBAR as rbar, KAPPA as kappa, SIGMA_R as sigma_r,
    BETA_ZR as beta_zr, SIGMA_Z as sigma_z, CBAR as Cbar,
    CHAT as Chat, I
)

## aim to output two T x N matrix of loadings

corr_zr = beta_zr / (sigma_z * sigma_r)
mean_exp_neg_beta = np.exp(-beta_star)/(1 - scale)

## function to compute derivatives of D and Jstar
r_pts = approx.r.values
n = len(r_pts)
def centered_diff(f):
    df = np.zeros(n)
    df[1:-1] = ((f(r_pts[2:]) - f(r_pts[:-2])) / (r_pts[2:] - r_pts[:-2])).flatten()
    df[0] = ((f(r_pts[1]) - f(r_pts[0])) / (r_pts[1] - r_pts[0])).item()
    df[-1] = ((f(r_pts[-2]) - f(r_pts[-1])) / (r_pts[-2] - r_pts[-1])).item()
    return df


# function to compute loadings based on Taylor
def loadings_Taylor(r, P, A, chi, sigmaj, corr_zj, T, N):
    
    # store interpolated functions of interest rates
    dD = interpolate.interp1d(r_pts, centered_diff(D), fill_value="extrapolate")
    dJstar = interpolate.interp1d(r_pts, centered_diff(Jstar), fill_value="extrapolate")

    B = P[:-1,:]/(Chat*I)
    rhat =  (1 - kappa) * rbar + kappa * r

    dD_pts = dD(rhat[:-1]).reshape([-1, 1]) * sigma_r
    D_hat_pts = D(rhat[:-1]).reshape([-1, 1])
    D_pts = D(r[:-1]).reshape([-1, 1])
    dJ_pts = dJstar(rhat[:-1]).reshape([-1, 1]) * sigma_r


    # adjust timing of assets in place
    A[0,:] = 0
    A[1:,:] = A[:-1,:]

    ## compute sum used in A_mu and A_xi
    summ = np.zeros((T, N)) 
    for t in np.arange(1,T):
        summ[t] = (chi[t-1, :t, :]*sigmaj[:t,:]*corr_zj[:t,:]*np.exp(-0.5*sigmaj[:t,:]**2)).sum(axis = 0)

    # A_mu/A_xi[t] is loadings at date t
    A_mu = pi*np.sqrt(1 - corr_zr**2)*summ/B
    A_xi = (dD_pts*(pi*A/(D_pts*Chat*I)) + dJ_pts + 
            mean_exp_neg_beta*dD_pts*(mean_exp_neg_beta*D_hat_pts >= 1/Chat) + 
            pi*corr_zr*summ)/B

    return A_mu, A_xi


# function to compute loadings based on projection
def loadings_projection(r, P, A, chi, sigmaj, corr_zj, T, N):
    
    D_pts = D(r[:-1]).reshape([-1,1])
    B = P[:-1,:]/(Chat*I)
    A[0,:] = 0 # adjust timing of assets in place
    A[1:,:] = A[:-1,:]

    # covariance terms - see sdf_compute.py for integrator function
    cov_Jstar_xi = integrator(Jstarvec, (r_adj - cond_mean)/sigma_r, r[:-1]).reshape([-1,1])
    cov_D_xi = integrator(Dvec,  (r_adj - cond_mean)/sigma_r, r[:-1]).reshape([-1,1])
    cov_optval_xi = integrator(optval/Chat,  (r_adj - cond_mean)/sigma_r, r[:-1]).reshape([-1,1])

    # compute sum used in both loadings
    summ = np.zeros((T, N)) 
    for t in np.arange(1,T):
        summ[t] = (chi[t-1, :t, :]*sigmaj[:t,:]*corr_zj[:t,:]).sum(axis = 0)

    # A_mu/A_xi[t] is loadings at date t
    A_mu = pi*np.sqrt(1 - corr_zr**2)*summ/B    
    A_xi = (cov_D_xi*(pi*A/(D_pts*Chat*I)) + 
            cov_Jstar_xi + 
            cov_optval_xi + 
            pi*corr_zr*summ)/B

    return A_mu, A_xi
