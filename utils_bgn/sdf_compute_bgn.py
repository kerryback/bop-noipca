import numpy as np
from scipy.integrate import quad
from scipy.stats import chi2
import pandas as pd
import os

import scipy.special
from scipy.sparse import csr_matrix, diags, kron

from scipy.stats import expon
from scipy.optimize import fsolve
from .vasicek import *
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from config import (
    PI as pi, RBAR as rbar, KAPPA as kappa, SIGMA_R as sigma_r,
    SIGMA_Z as sigma_z, CBAR as Cbar, CHAT as Chat,
    GAMMA_GRID as gamma_grid, I
)

# Get path to solution files relative to this module
_SOLFILES_DIR = os.path.join(os.path.dirname(__file__), 'BGN_solfiles')

Chat = np.exp(Cbar)
approx = pd.read_csv(os.path.join(_SOLFILES_DIR, 'Jstar.csv'))
Jstar = interpolate.interp1d(approx.r, approx.J, fill_value="extrapolate")
r_pts = np.array(approx.r).reshape(len(approx.r), 1)

## makes a big computational difference to just approximate D with a fine linear interpolation
D = interpolate.interp1d(approx.r, np.vectorize(D)(approx.r), fill_value="extrapolate")

# cond mean of rf rate at t+1 on grid
cond_mean = kappa * r_pts + (1 - kappa) * rbar  # Mean

n_pts = 100  # number of GH pts
rshock_nodes, wgts = np.polynomial.hermite.hermgauss(n_pts)

# adjust GH nodes - gives a len(r_pts) x n_pts matrix of pts
# each column corresponds to a value r[t]
# each row corresonds to values of r[t+1] given the initial r[t]
r_adj = np.sqrt(2) * sigma_r * rshock_nodes.reshape(1, n_pts) + cond_mean

# useful to only evaluate these in advance
Dvec = D(r_adj)
Jstarvec = Jstar(r_adj)


# this function computes integrand of E[(Chat D[r_t+1] e^{-beta} - 1)+] conditional on r - useful below
def integrand_optval(r):
    Dvec = D(r)
    mini = np.minimum(np.log(Chat * Dvec), beta_star)
    mini = np.where(Dvec > 0, mini, 0)
    return np.where(
        Dvec > 0,
        np.exp(-beta_star / scale)
        * (
            Chat / (1 - scale) * Dvec * np.exp(mini * (1 / scale - 1))
            - np.exp(mini / scale)
        ),
        0,
    )


optval = integrand_optval(r_adj)


# function to compute integrand of integral (1)
def integrand_optval_sq(r):
    Dvec = D(r)
    mini = np.minimum(np.log(Chat * Dvec), beta_star)
    mini = np.where(Dvec > 0, mini, 0)
    return np.where(
        Dvec > 0,
        np.exp(-beta_star / scale)
        * (
            Chat**2 / (1 - 2 * scale) * Dvec**2 * np.exp(mini * (1 / scale - 2))
            - 2 * Chat / (1 - scale) * Dvec * np.exp(mini * (1 / scale - 1))
            + np.exp(mini / scale)
        ),
        0,
    )


optval_sq = integrand_optval_sq(r_adj)


### Now start computing integrals


## computes integral of f(r_t+1) g(r_t+1) dr_t+1
# at a vector of r_t called rdata
# takes as input f, g evaluated at r_adj
def integrator(fpts, gpts, rdata):
    integ = (wgts.reshape(1, n_pts) * fpts * gpts).sum(axis=1) / np.sqrt(np.pi)
    interp = interpolate.interp1d(approx.r, integ, fill_value="extrapolate")
    return interp(rdata)


## integrals (6),(7), (10) - compute on a r by y grid
def integ_expy(f, y):
    # Adjust r based on y
    r_adj = sigma_r * (
        np.sqrt(2) * rshock_nodes.reshape(1, n_pts, 1) + y[None, None, :]
    ) + cond_mean.reshape(len(r_pts), 1, 1)
    # r_adj shape: (len(r_pts), n_pts, len(y))

    fvec = f(r_adj)  # Shape: (len(r_pts), n_pts, len(y))
    wgts_expanded = wgts[None, :, None]  # Reshape wgts - Shape: (1, n_pts, 1)

    # Perform the integration, ensuring alignment - integ is shape (len(r_pts), len(y))
    integ = (
        (wgts_expanded * fvec).sum(axis=1)
        * np.exp(y[None, :] ** 2 / 2)
        / np.sqrt(np.pi)
    )

    # Output interpolant
    return RegularGridInterpolator(
        (r_pts.reshape(-1), y), integ, bounds_error=False, fill_value=None
    )


# min and max values are determined by max/min of a simulated panel +- some error for buffer
y_pts = np.linspace(-0.5, 0.5, 500)
interp_integ_Dexpy = integ_expy(D, y_pts)
interp_integ_Jsexpy = integ_expy(Jstar, y_pts)
interp_integ_optvalexpy = integ_expy(integrand_optval, y_pts)


def sdf_compute(N, T, arr_tuple):
    r, mu, xi, sigmaj, chi, beta, corr_zj, eret, ret, P, corr_zr, book, op_cash_flow, loadings_mu_taylor, loadings_xi_taylor, loadings_mu_proj, loadings_xi_proj  = arr_tuple

    # precompute some of the integrals to be used below

    # expected value of J[r_{t+1}^2] cond on dates 1, ..., T-1
    data_integ_J_sq = integrator(Jstarvec, Jstarvec, r[1:-1])
    data_integ_JD = integrator(Jstarvec, Dvec, r[1:-1])
    data_integ_D_sq = integrator(Dvec, Dvec, r[1:-1])
    data_integ_J_optval = integrator(Jstarvec, optval, r[1:-1])
    data_integ_D_optval = integrator(Dvec, optval, r[1:-1])
    data_integ_optval_cross = integrator(optval, optval, r[1:-1])
    data_integ_optval_sq = integrator(optval_sq, 1, r[1:-1])

    # precompute the simplest terms to be used below
    term1 = I**2 * Chat**2 * data_integ_J_sq
    term2 = I**2 * data_integ_optval_cross

    exp_beta = np.exp(-beta)
    exp_corr_zr = np.exp(-0.5 * sigmaj**2 * corr_zj**2 * corr_zr**2)

    def sdf_loop(t, iter):
        # compute sparse matrices to be used repeatedly
        # chi_{s, t}^i for t going from date 1, ..., T-1, s going from dates 1 to t,
        # and i going from 1 to N
        chisp = csr_matrix(chi[t, : t + 1, :])
        col1 = chisp.multiply(exp_beta[: t + 1, :])
        col2 = chisp.multiply(sigmaj[: t + 1, :] * corr_zj[: t + 1, :])
        col3 = chisp.multiply(exp_corr_zr[: t + 1, :])

        # compute integrals of the form E_t f(r_{t+1}) e^{y xi_{t+1}}
        integ_y = col2.copy() * corr_zr
        rdata = np.full_like(integ_y.data, r[t + 1])
        integ_Dexpy_data = integ_y.copy()
        integ_Jsexpy_data = integ_y.copy()
        integ_optvalexpy_data = integ_y.copy()
        integ_Dexpy_data.data = interp_integ_Dexpy((rdata, integ_y.data))
        integ_Jsexpy_data.data = interp_integ_Jsexpy((rdata, integ_y.data))
        integ_optvalexpy_data.data = interp_integ_optvalexpy((rdata, integ_y.data))

        # generate terms corresponding to ERiRj, i != j
        result3 = kron(col1, col1)
        term3 = (
            (Chat * I * pi) ** 2
            * result3.sum(axis=0).reshape(N, N).A
            * data_integ_D_sq[t]
        )

        result4 = kron(col2, col2).copy()
        result4.data = np.exp(result4.data)  # only exponentiate non-zero entries
        term4 = (Chat * I * pi) ** 2 * result4.sum(axis=0).reshape(N, N).A

        result5 = kron(col1, col3.multiply(integ_Dexpy_data))
        term5 = (Chat * I * pi) ** 2 * result5.sum(axis=0).reshape(N, N).A

        term6 = 2 * I**2 * Chat * data_integ_J_optval[t]
        term7 = (
            I**2
            * Chat
            * pi
            * np.array(
                np.sum(col1, axis=0)
                * (Chat * data_integ_JD[t] + data_integ_D_optval[t])
            )
        )
        term8 = (
            I**2
            * Chat
            * pi
            * np.array(
                np.sum(
                    col3.multiply(Chat * integ_Jsexpy_data + integ_optvalexpy_data),
                    axis=0,
                )
            )
        )

        # make corrections to these terms when i = j
        term2t = term2[t] + np.diag(
            I**2
            * (data_integ_optval_sq[t] - data_integ_optval_cross[t])
            * np.ones((N,))
        )

        diag3_sub = np.diag(
            np.array(
                (
                    (Chat * I * pi) ** 2
                    * data_integ_D_sq[t]
                    * chi[t, : t + 1, :]
                    * exp_beta[: t + 1, :] ** 2
                ).sum(axis=0)
            ).reshape(
                N,
            )
        )
        term3 = term3 - diag3_sub + diag3_sub / pi

        diag4_sub = np.diag(
            np.array(
                (
                    (Chat * I * pi) ** 2
                    * chi[t, : t + 1, :]
                    * np.exp(sigmaj[: t + 1, :] ** 2 * corr_zj[: t + 1, :] ** 2)
                ).sum(axis=0)
            ).reshape(
                N,
            )
        )
        diag4_add = np.diag(
            np.array(
                (
                    (Chat * I) ** 2
                    * pi
                    * chi[t, : t + 1, :]
                    * np.exp(sigmaj[: t + 1, :] ** 2)
                ).sum(axis=0)
            ).reshape(
                N,
            )
        )
        term4 = term4 - diag4_sub + diag4_add

        diag5_sub = (Chat * I * pi) ** 2 * np.diag(
            np.array(
                (col1.multiply(col3).multiply(integ_Dexpy_data)).sum(axis=0)
            ).reshape(
                N,
            )
        )
        term5 = term5 - diag5_sub + diag5_sub / pi

        # E_t{R_t+1^i R_t+1^j}, for t corresponding to dates 1, ..., T-1
        ER = np.zeros((N + 1, N + 1))

        # aggregate terms and store matrix
        ER[1:, 1:] = (
            term1[t]
            + term2t
            + term3
            + term4
            + term5
            + term5.T
            + term6
            + term7
            + term7.T
            + term8
            + term8.T
        )
        ER[1:, 1:] /= np.outer(P[t + 1], P[t + 1])

        # store when at least one of assets in risk free
        ER[0, 0] = np.exp(2 * r[t + 1])
        ER[0, 1:] = (eret[t + 1, :] + 1) * np.exp(r[t + 1])
        ER[1:, 0] = (eret[t + 1, :] + 1) * np.exp(r[t + 1])

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
                port = scipy.linalg.solve(ER, np.ones((N + 1, 1)), assume_a="pos").reshape(-1)
            except Exception as e:
                # Handle failure after retry
                print(f"Second attempt failed: {e}. Using fallback value for port.")
                port = np.full((N+1, ), np.nan)


        port /= port.sum()
        sdf_ret = -(port[1:] * (1 + ret[t + 1, :] - np.exp(r[t + 1]))).sum()


        cond_var = ER[1:, 1:] - np.outer(
            1 + eret[t + 1, :], 1 + eret[t + 1, :]
        )  # conditional variance of risky assets
        max_sr = -(port[1:] * (1 + eret[t + 1, :] - np.exp(r[t + 1]))).sum() / np.sqrt(
            port[1:] @ (cond_var @ port[1:])
        )
        return (
            # -port[1:],
            sdf_ret,
            max_sr,
            1 + eret[t + 1, :] - np.exp(r[t + 1]),
            cond_var,
        )

    return sdf_loop
