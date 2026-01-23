
import numpy as np
import pandas as pd
import os
from scipy import interpolate
from config import (
    KP14_BURNIN as burnin, KP14_DT as dt,
    KP14_MU_X as mu_x, KP14_MU_Z as mu_z,
    KP14_SIGMA_X as sigma_x, KP14_SIGMA_Z as sigma_z,
    KP14_THETA_EPS as theta_eps, KP14_SIGMA_EPS as sigma_eps,
    KP14_THETA_U as theta_u, KP14_SIGMA_U as sigma_u,
    KP14_DELTA as delta, KP14_MU_LAMBDA as mu_lambda,
    KP14_SIGMA_LAMBDA as sigma_lambda,
    KP14_MU_H as mu_H, KP14_MU_L as mu_L,
    KP14_LAMBDA_H as lambda_H, KP14_LAMBDA_L as lambda_L,
    KP14_R as r, KP14_GAMMA_X as gamma_x, KP14_GAMMA_Z as gamma_z,
    KP14_ALPHA as alpha, KP14_PROB_H as prob_H,
    KP14_A_0 as A_0, KP14_A_1 as A_1, KP14_A_2 as A_2, KP14_A_3 as A_3,
    KP14_RHO as rho, KP14_C as C
)

# Define A function (from parameters_kp14.py)
A = lambda ep, u: (A_0 + (ep - 1) * A_1 + (u - 1) * A_2 + (ep - 1) * (u - 1) * A_3)

from .sdf_compute_kp14 import *
from .loadings_compute_kp14 import *

# Get path to solution files relative to this module
_SOLFILES_DIR = os.path.join(os.path.dirname(__file__), 'KP14_solfiles')

# script to simulate panel for discretized Kogan Papanikolaou (2014) model

def create_arrays(N, T):
    # read in G functions estimated in kp14_fd.py
    # recall they don't include lambda_f, which varies across firms
    G_in = pd.read_csv(os.path.join(_SOLFILES_DIR, 'G_func.csv'))
    eps_grid = G_in.eps.values
    G_up = interpolate.interp1d(eps_grid, G_in.G_up.values, fill_value="extrapolate")
    G_down = interpolate.interp1d(eps_grid, G_in.G_down.values, fill_value="extrapolate")

    # Load integrals file
    integ_data = np.load(os.path.join(_SOLFILES_DIR, 'integ_results.npz'))

    # Access some of the integrals
    A_mod_lst = integ_data["A_mod_lst"]
    G_up_lst = integ_data["G_up_lst"]
    G_down_lst = integ_data["G_down_lst"]

    Et_A_mod = interpolate.interp1d(eps_grid, A_mod_lst, fill_value="extrapolate")
    Et_G_up = interpolate.interp1d(eps_grid, G_up_lst, fill_value="extrapolate") # again - be sure to adjust any G term by lambda_f
    Et_G_down = interpolate.interp1d(eps_grid, G_down_lst, fill_value="extrapolate")

    # simulates length x N matrix of Cox-Ingersoll-Ross processes
    # each column is a separate process for each firm
    # dr = kappa(theta - r) dt + sigma dB
    def sim_cir(kappa, theta, sigma, length = T+1, init = None):

        r = np.zeros((length, N))
        # Draw r0 from stationary Gamma distribution
        if init == None:
            shape = 2 * kappa * theta / sigma ** 2
            scale = sigma ** 2 / (2 * kappa)
            r[0,:] = np.random.gamma(shape, scale, size = (N,))
        else:
            r[0,:] = 1

        c = (sigma ** 2 * (1 - np.exp(-kappa * dt))) / (4 * kappa)
        d = 4 * kappa * theta / sigma ** 2
        for t in range(1, length):
            lambda_ = (4 * kappa * np.exp(-kappa * dt) * r[t - 1, :]) / (sigma ** 2 * (1 - np.exp(-kappa * dt)))
            z = np.random.noncentral_chisquare(df=d, nonc=lambda_)
            r[t, :] = c * z

        return r

    # simulates length T+1 vector of GBM
    # dS/S = mu dt + sigma dB
    def sim_gbm(mu, sigma, S0):
        S = np.zeros((T+1, ))
        S[0] = S0

        # Simulate Brownian increments
        dW = np.random.normal(0, np.sqrt(dt), size=T)
        W = np.cumsum(dW)  
        S[1:] = S0 * np.exp((mu - 0.5 * sigma**2) * np.arange(1, T+1)*dt + sigma * W)

        return S.reshape([-1, 1])

    # simulates (T+1)xN matrix of regime switching project rates
    # and simulates TXN matrix of project arrivals
    # each column corresponds to a firm n with average rate lambda_f[n]
    def sim_arrivals(lambda_f):

        state = np.zeros((T+1, N))
        state[0,:] = np.random.binomial(1, prob_H, size = (N, ))  # 1 = high, 0 = low
        for t in range(1, T + 1):
            curr = state[t - 1,:]
            switch_prob = np.where(curr == 1, mu_H * dt, mu_L * dt)
            state[t,:] =  np.where(np.random.random((N,)) < switch_prob, 1 - curr, curr)

        rate = lambda_L + state*(lambda_H - lambda_L)
        
        rate*= lambda_f

        arrivals = np.zeros((T+1, N))
        arrivals[0,:] = 0
        arrivals[1:,:] = (np.random.random((T, N)) < dt*rate[:-1,:]).astype(int)

        return arrivals, state

    # lambda_f[n] is (average) project arrival for firm n
    lambda_f = mu_lambda*delta - sigma_lambda*delta*np.log(np.random.uniform(size = (N,)))

    # eps[t, n] is value of eps shock at time t for firm n
    eps = sim_cir(theta_eps, 1, sigma_eps)

    # x[t], z[t] are x[t], z[t] shocks at time t
    x = sim_gbm(mu_x, sigma_x, 0.5) # not sure what initial value should be
    z = sim_gbm(mu_z, sigma_z, 0.5)

    # arrivals[t, n] is whether a project arrives at date t for firm n
    # high[t, n] is whether the arrival rate of projects for firm n are in high state at date t
    arrivals, high = sim_arrivals(lambda_f)

    # chi[t1, t2, n] is whether project initiated at time t1 is alive at time t2 for firm n
    chi = np.zeros((T + 1, T + 1, N)) 
    chi[range(T+1), range(T+1),:] = arrivals
    chi = np.cumsum(chi, axis = 1)

    # alive[t1, t2, n] is equal to 0 if at time t2 the project which arrived at date t1 of firm n is dead 
    alive = np.ones((T + 1, T + 1, N))
    alive[chi == 1] = (np.random.random((T+1, T+1, N)) >= dt*delta)[chi == 1] 
    alive[:, 1:,:] = np.cumprod(alive[:, :-1, :], axis = 1)  # shift is because projects can't be dead in first period they exist

    chi = chi* alive # adjust chi for project death

    # Kj[t1, k] is capital allocated at time t1 by firm k if a project were to be initialized
    Kj = ((alpha*z*A(eps, 1))**(1/(1 - alpha))) 

    # K[t1, t2, n] is capital allocated at t2 to project arriving at time t1 for firm n (0 if project is dead)
    K = chi*Kj[:, np.newaxis, :] 

    # book[t, n] is the book value of firm n at date t (see footnote 12 in KP14)
    book = x/z*K.sum(axis = 0)

    # uj[t1, t2, n] is uj process at time t2 for project arriving at time t1 for firm n
    uj = np.zeros((T+1, T+1, N)) 
    for t in range(T+1):
        uj[t, t:, :] = sim_cir(theta_u, 1, sigma_u, length = T +1 - t, init = 1)
        
    # cashflow[t1, t2, n] is cashflow at time t2 from project arriving at t1 for firm n
    # cashflow from t to t + 1 is flow_t dt, arriving at date t + 1
    cashflow = np.zeros((T+1, T+1, N))
    cashflow[:, 1:, :] = eps[np.newaxis, :-1, :]*x[np.newaxis, :-1]*uj[:, :-1,:]*K[:, :-1,:]**alpha* dt 
    op_cashflow = cashflow # cashflow from operations
    op_cashflow = np.sum(op_cashflow, axis = 0)
    cashflow[range(T+1), range(T+1),:] = -K[range(T+1), range(T+1),:]*x/z

    # cashflow is overwritten by sum_t1 cashflow[t1, t2, n]
    # cashflow[t2, n] is total cashflows from all projects at time t2 for firm n
    cashflow = np.sum(cashflow, axis = 0)

    # VAP[t1, t2, n] is value at time t2 of project arriving at t1 for firm n
    VAP = np.zeros((T+1, T+1, N))
    VAP = x[np.newaxis, :]*K**alpha * A(eps[np.newaxis, :, :], uj)

    # VAP is overwritten by sum_t1 VAP[t1, t2, n]
    # VAP[t2, n] is total value of assets in place at time t2 for firm n
    VAP = np.sum(VAP, axis = 0)

    # PVGO[t, n] is PVGO for firm n at date t
    # recall G_func.csv doesn't include lambda_f
    PVGO = z**(alpha/(1 - alpha))*x*lambda_f*(G_down(eps)*(high == 0) + G_up(eps)*(high == 1)) 

    # price[t] is price at date t
    price = VAP + PVGO

    # rets[t] is returns from date t to t+1
    rets = (price[1:,:] + cashflow[1:,:]) /price[:-1,:] - 1

    # erets[t, n] is E_t[R_{t+1}] for firm n
    # compute 4 terms from the KP14.tex overleaf
    Et_x = np.exp(mu_x*dt)*x
    alph = alpha/(1 - alpha)
    Et_z_alph = z**alph*np.exp(alph*mu_z*dt + 0.5*alph*(2*alpha-1)/(1 - alpha)*sigma_z**2*dt)

    EtA = A(1+ (eps[np.newaxis, :, :] -1)*np.exp(-theta_eps*dt), 1 + (uj - 1)*np.exp(-theta_u*dt))
    term1 = Et_x*(1 - delta*dt)*np.sum(chi*EtA*K**alpha, axis = 0)

    Et_G = ((high == 0)*lambda_f*((1 - mu_L*dt)* Et_G_down(eps) + mu_L*dt* Et_G_up(eps)) + 
        (high == 1)*lambda_f*((1 - mu_H*dt)* Et_G_up(eps) + mu_H*dt * Et_G_down(eps)))

    term2 = Et_z_alph*Et_x*Et_G
    term3 = np.sum(chi*eps[np.newaxis, :, :]*uj*x[np.newaxis, :]*K**alpha*dt, axis = 0)

    rate = lambda_f*(lambda_L + (lambda_H - lambda_L)*high)
    term4 = rate*dt*C*Et_z_alph*Et_x*Et_A_mod(eps)

    erets = (term1 + term2  + term3 + term4) /price - 1

    loadings_z_taylor, loadings_x_taylor = loadings_Taylor(K, x, z, eps, uj, rate, high, lambda_f, price, G_up, G_down)
    loadings_z_proj, loadings_x_proj = loadings_projection(K, x, z, eps, uj, rate, high, lambda_f, price, erets, Et_G_up, Et_G_down, Et_A_mod)


    arr_tuple = K, book, op_cashflow, x, z, eps, uj, chi, rate, high, Et_G, EtA, alph, Et_z_alph, price, rets, erets, lambda_f, loadings_z_taylor, loadings_x_taylor, loadings_z_proj, loadings_x_proj 
    
    return arr_tuple


def create_panel(N, T, arr_tuple):

    K, book, op_cashflow, x, z, eps, uj, chi, rate, high, Et_G, EtA, alph, Et_z_alph, P, ret, eret, lambda_f, loadings_z_taylor, loadings_x_taylor,  loadings_z_proj, loadings_x_proj = arr_tuple
    
    df = pd.DataFrame(
        {
            "firmid": np.repeat(range(N), T),
            "month": np.tile(range(T), N),
          
            # market and book are at dates 0, ..., T-1
            "mve": np.reshape(P[:-1,:].T, (N * T,)),
            "book": np.reshape(book[:-1,:].T, (N * T)),
            "op_cash_flow": np.reshape(
                np.concatenate((np.zeros((1, N)), op_cashflow[:-2,:])).T, (N * T,)
            ),
            "ret": (ret.T).reshape(
                N * T,
            ),
            # "A_1_taylor": (loadings_z_taylor[:-1,:].T).reshape(
            #     N * T,
            # ),
            # "A_2_taylor": (loadings_x_taylor[:-1,:].T).reshape(
            #     N * T,
            # ),
            # "A_1_proj": (loadings_z_proj[:-1,:].T).reshape(
            #     N * T,
            # ),
            # "A_2_proj": (loadings_x_proj[:-1,:].T).reshape(
            #     N * T,
            # )
        }
    )
    df.set_index(["firmid", "month"], inplace=True)

    # roe = cash flow over beginning of month book equity
    # roe at date 0 = NAN
    # roe at date 1 = cash flow from 0 to 1 divided by date 0 book equity
    # If book equity is zero or negative, set roe to NaN (will be dropped later)
    df["roe"] = df.groupby("firmid", group_keys=False).apply(
        lambda d: pd.Series(
            np.where(d.book > 0, d.op_cash_flow / d.book, np.nan),
            index=d.index
        ).shift()
    )

    df["bm"] = df.book / df.mve
    df["cumret"] = df.groupby("firmid", group_keys=False).ret.apply(
        lambda x: (1 + x).cumprod()
    )
    df["mom"] = df.groupby("firmid", group_keys=False).cumret.apply(
        lambda x: x.shift(2) / x.shift(13) - 1
    )
    # Asset growth: set to NaN when previous book equity is zero or negative (will be dropped later)
    df["agr"] = df.groupby("firmid", group_keys=False).book.apply(
        lambda x: pd.Series(
            np.where(x.shift(1) > 0, x.pct_change(), np.nan),
            index=x.index
        )
    )
    df.index = df.index.swaplevel()
    df.sort_index(level=["month", "firmid"], inplace=True)
    df = df.drop(columns=["book", "cumret", "op_cash_flow"])  
    df.reset_index(inplace=True)
    df.ret -= (np.exp(r*dt) - 1)
    df = df.rename(columns={"ret": "xret"})

    df = df[df.month > burnin - 1]
    return df


#N, T = 100, burnin+10
#arr_tuple = create_arrays(N, T)
#print(create_panel(N, T, arr_tuple).xret.describe())