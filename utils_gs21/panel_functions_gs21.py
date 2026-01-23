
import numpy as np
import pandas as pd
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
from .sdf_compute_gs21 import *
from .loadings_compute_gs21 import *

# function to generate Gomes Schmid 2021 panel with N firms and T periods
def create_arrays(N, T):

    def simulate_ar1(nfirms = N, length = T):

        # Pre-allocate
        x = np.empty(length+1, dtype=float)
        z = np.empty((length+1, nfirms), dtype=float)

        # Stationary variances
        var_x = sigma_x**2 / (1.0 - rho_x**2)
        var_z = sigma_z**2 / (1.0 - rho_z**2)

        # Initialize at steady state (t=0)
        x[0] = np.random.normal(xbar, np.sqrt(var_x))
        z[0, :] = np.random.normal(zbar, np.sqrt(var_z), size=nfirms)

        # Innovations
        eps = np.random.standard_normal(length)
        epsj = np.random.standard_normal((length, nfirms))

        # Simulate forward
        for t in range(1, length+1):
            x[t] = xbar + rho_x * (x[t-1] - xbar) + sigma_x * eps[t-1]
            z[t, :] = zbar + rho_z * (z[t-1, :] - zbar) + sigma_z * epsj[t-1, :]

        return x, z
    
    # x is length T+1 vector of aggregate productivity shocks
    # z is shape (T+1) by N vector of idiosyncratic shocks
    x, z = simulate_ar1()
    eta = (np.random.rand(T+1, N) < zeta).astype(int)

    # i_cost is shape (T+1) by N vector of realized investment costs
    i_cost =  np.random.uniform(imin, imax, size=(T+1, N))


    k = np.ones((T+1, N))                       # capital stock
    b = np.zeros((T+1, N))                      # debt/capital stock
    invest = np.zeros((T+1, N))                 # investment decision dummies
    cutoff = np.zeros((T+1, N))                 # endogenous cutoff (invest <=> i_cost <= cutoff)
    P = np.zeros((T+1, N))                      # prices before investment shock is realized (equation 11)
    default = np.zeros((T+1, N), dtype = bool)  # indicator for whether a firm defaulted (and hence characteristics correspond to replacement firm)
    for t in range(T+1):

        xt = x[t]*np.ones(z[t].shape)

        ##### compute prices and adjust for default #######
        

        P[t] = ((eta[t] == 0)*P_down((z[t], xt,  b[t])) +
                (eta[t] == 1)*P_up(( z[t], xt, b[t])) )
        
        # adjust everything for default
        default[t] = (P[t] <= 0)
        while (P[t] <= 0).sum() > 0: # GS21.m sets P = 0 if default occurs
            defs = (P[t]<=0)
            ndefaults = np.sum(defs)

            if ndefaults < N:
                b[t][defs] = 0 # initialize new firms with 0 debt
                k[t][defs] = alpha*np.mean(k[t][~defs]) # initialize with alpha*average capital stock
                z[t:, defs] = simulate_ar1(ndefaults, T-t)[1] # simulate new shocks
            else:
                b[t][defs] = 0 # initialize new firms with 0 debt
                k[t][defs] = alpha # initialize with alpha*average capital stock
                z[t:, defs] = simulate_ar1(ndefaults, T-t)[1] # simulate new shocks
                xt = np.random.normal(xbar, np.sqrt(sigma_x**2 / (1.0 - rho_x**2)))

            # now input prices for replacement firms 
            P[t] = ((eta[t] == 0)*P_down((z[t], xt,  b[t])) +
                    (eta[t] == 1)*P_up(( z[t], xt, b[t])) )
        

        ##### compute realized decisions and endogenous states ####
        # investment cutoff conditional on eta
        cutoff[t] = (eta[t]== 0)*i_cut_down((z[t], xt, b[t])) + (eta[t] == 1)*i_cut_up((z[t], xt, b[t]))
        invest[t] = (i_cost[t] <= cutoff[t])

        # pre-compute debt if re-financing occurs
        b_new_no = b_refin_0((z[t], xt, b[t])) # debt if no investment 
        b_new_I = b_refin_I((z[t], xt, b[t])) # debt if there is investment

        
        # update debt based on refinancing and investment decisions
        if t < T:
            b[t+1,:] = ((eta[t] == 0)*((1-invest[t])*b[t,:] + invest[t]*b[t,:]/g)     # no refinancing
                        + (eta[t] == 1)*((1-invest[t])*b_new_no + invest[t]*b_new_I)) # refinancing
            k[t+1,:] = (1 + invest[t]*(g-1))*k[t,:]


        
    ##### compute returns after computing cash flows ######
    xfull = x.reshape(-1, 1)*np.ones_like(z)
    # compute cashflows contingent on eta and investment decision
    prof_I_down = (((1 - tau)*np.exp(xfull + z) - delta) - (1 - tau)*b)                      # invest and eta = 0
    prof_0_down = (((1 - tau)*np.exp(xfull + z) - delta) - (1 - tau)*b)                             # no invest and eta = 0
    prof_I_up = (((1 - tau)*np.exp(xfull + z) - delta) - (1 - tau)*b +                       # invest and eta = 1
                 (1 - kappa_b)*Q_I((z, xfull, b_refin_I((z, xfull, b)))) - Q_I((z, xfull, b/g)))
    prof_0_up = (((1 - tau)*np.exp(xfull + z) - delta) - (1 - tau)*b +                              # no invest and eta = 1
                    (1 - kappa_b)*Q_0((z, xfull, b_refin_0((z, xfull, b)))) - Q_0((z, xfull, b)))
    
    
    
    # compute expected CF before investment cost
    prob_i = cutoff/(imax - imin)
    mean_i_cost = (cutoff + imin)/2
    Ecashflow = (prob_i*(eta == 0)*(prof_I_down  - mean_i_cost) +
                (1-prob_i)*(eta == 0)*prof_0_down +  
                prob_i*(eta == 1)*(prof_I_up  - mean_i_cost) +  
                (1-prob_i)*(eta == 1)*prof_0_up)
    P_ex = P - Ecashflow
    # put lower bound on P
    P_ex = np.maximum(0.05/k, P_ex)

    op_cashflow = (1 - tau)*np.exp(xfull + z) - delta
        
    # rets is T by N
    # no ret in final period
    
    #rets = ((k[:-1,:]*cashflow[:-1,:]*np.exp(r) + k[1:,:]*P[1:,:])/(k[:-1,:]*P[:-1,:])*(default[1:,:] == 0) + 
    #        (k[:-1,:]*cashflow[:-1,:]*np.exp(r))/(k[:-1,:]*P[:-1,:])*(default[1:,:] == 1) - 1)

    rets = (k[1:,:]*P[1:,:]/(k[:-1,:]*P_ex[:-1,:])*(default[1:,:] == 0) + 
            0*(default[1:,:] == 1) - 1)

    # truncating returns
    # rets = np.maximum(-1, np.minimum(rets, 2))
    
    # function to compute expected returns at a given point in time
    # uses GH quadrature to integrate over realizations of z and x
    def integ_zi_x(icutoff, eta, bi, zi, x, n = 10):
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


        # compute todays profits when eta = 0 (down), eta = 1 (up)
        # and when investment occurs (I), or no investments (0)
        '''
        prof_I_down = (((1 - tau)*np.exp(x + zi) - delta)-(icutoff + imin)/2 - (1 - tau)*bi)
        prof_0_down = (((1 - tau)*np.exp(x + zi) - delta) - (1 - tau)*bi)
        prof_I_up = (((1 - tau)*np.exp(x + zi) - delta)-(icutoff + imin)/2 - 
                     (1 - tau)*bi + (1 - kappa_b)*Q_I((zi, x, b_refin_I((zi, x, bi)))) - Q_I((zi, x, bi/g)))
        prof_0_up = (((1 - tau)*np.exp(x + zi) - delta) - (1 - tau)*bi + 
                     (1 - kappa_b)*Q_0((zi, x, b_refin_0((zi, x, bi)))) - Q_0((zi, x, bi)))
       
        # compute cashflow, integrating over investment decision
        cashflow = ((icutoff - imin)/(imax - imin)*(eta == 0)*prof_I_down*np.exp(r) +  
                    (imax - icutoff)/(imax - imin)*(eta == 0)*prof_0_down*np.exp(r) +  
                    (icutoff - imin)/(imax - imin)*(eta == 1)*prof_I_up*np.exp(r)  +  
                    (imax - icutoff)/(imax - imin)*(eta == 1)*prof_0_up*np.exp(r))
        '''
               
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
        exp_P = (w * (refin + no_refin)).sum(axis=1) / np.pi # (normalize by pi since it's a double integral)

        return exp_P # cashflow.reshape(-1) + 
    
    ##### compute expected returns #####
    # erets is shape (T+1) by N matrix of expected returns
    erets = np.zeros((T+1, N))
    for t in range(T+1):
        erets[t] = integ_zi_x(cutoff[t], eta[t], b[t], z[t], x[t], n = 10)/P_ex[t] -1

    
    
    loadings_x_taylor = loadings_Taylor(k, b, x, z, cutoff, eta, P, P_ex, erets, rets, op_cashflow, default, N)
    loadings_x_proj = loadings_projection(k, b, x, z, cutoff, eta, P, P_ex, erets, rets, op_cashflow, default, N, T)

    arr_tuple = k, b, x, z,cutoff, eta, P, P_ex, erets, rets, op_cashflow, default, loadings_x_taylor, loadings_x_proj 
    #sdf_loop = sdf_compute(N, T, arr_tuple)
    #var = sdf_loop(10)[-1]
    #print(np.mean(invest))
    #print(erets.mean())
    #print(rets.mean())
    return arr_tuple




def create_panel(N, T, arr_tuple):

    book, debt, x, z,cutoff, eta, P, P_ex, erets, rets, op_cashflow, default, A_x_taylor, A_x_proj = arr_tuple
    
    P = P_ex
    df = pd.DataFrame(
        {
            "firmid": np.repeat(range(N), T),
            "month": np.tile(range(T), N),
          
            # market and book are at dates 0, ..., T-1
            "mve": np.reshape(book[:-1,:].T*P[:-1,:].T, (N * T,)),
            "default": np.reshape(default[:-1,:].T, (N * T,)), # default is a dummy that denotes the first month of a new firm (previous one defaulted)
            "book": np.reshape(book[:-1,:].T, (N * T)),
            "op_cash_flow": np.reshape(book[:-1,:].T*op_cashflow[:-1,:].T, (N * T)),
            "book_lev": np.reshape(debt[:-1,:].T, (N * T)),
            "ret": (rets.T).reshape(N * T),
            # "A_1_taylor": (A_x_taylor[:-1,:].T).reshape(
            #     N * T,
            # ),
            # "A_1_proj": (A_x_proj[:-1,:].T).reshape(
            #     N * T,
            # )
        }
    )
    
    df.set_index(["firmid", "month"], inplace=True)

    # determine if new firms have 12 months of data
    df['default_lag'] = (
        df.groupby('firmid')['default']
        .rolling(window=12, min_periods=12)
        .max()                 # 1 if any default happened in window
        .reset_index(level=0, drop=True)
        .astype(bool)          # ensure boolean dtype
    )

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
    df["roe"] = (1 - df["default"])*df['roe'] # roe is 0 in first month of new firm

    df["bm"] = df.book / df.mve
    df["mkt_lev"] = df["book_lev"] * df["bm"]  # Market leverage = book leverage * book-to-market
    df["cumret"] = df.groupby("firmid", group_keys=False).ret.apply(
        lambda x: (1 + x).cumprod()
    )
    df["mom"] = df.groupby("firmid", group_keys=False).cumret.apply(
        lambda x: x.shift(2) / x.shift(13) - 1
    )
    df["mom"] = (1 - df["default_lag"])* df["mom"] # you need past 12 months of data to compute momentum
    # Asset growth: set to NaN when previous book equity is zero or negative (will be dropped later)
    df["agr"] = df.groupby("firmid", group_keys=False).book.apply(
        lambda x: pd.Series(
            np.where(x.shift(1) > 0, x.pct_change(), np.nan),
            index=x.index
        )
    )
    df["agr"] = (1 - df["default"])*df['agr'] # agr is 0 in first month of new firm
    df.index = df.index.swaplevel()
    df.sort_index(level=["month", "firmid"], inplace=True)
    df = df.drop(columns=["book", "cumret", "op_cash_flow"])  
    df.reset_index(inplace=True)
    df.ret -= (np.exp(r) - 1)
    df = df.rename(columns={"ret": "xret"})

    df = df[df.month > burnin - 1]
    return df

#N, T = 200, burnin+720
#arr_tuple = create_arrays(N, T)
#print(create_panel(N, T, arr_tuple).xret.describe())