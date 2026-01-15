import numpy as np
import pandas as pd 
from joblib import Parallel, delayed
import scipy.linalg as linalg
from sklearn.linear_model import Ridge
from parameters import *

# calculate fama-french weights for a single month
def fama_french(data, chars, **kwargs):

    factor_dct = {}

    if len(chars) == 5:
        names = ["smb", "hml", "cma", "rmw", "umd"]
    else:
        names = ["smb", "hml", "umd"]
    name_dct = dict(zip(chars, names))

    # sort on size
    big = data["size"] > data["size"].median()  
    small = 1 - big
    mve = kwargs["mve"]
    
    for char in [c for c in chars if c != "size"]:

        # sort on characteristic
        low = data[char] <= data[char].quantile(0.3)
        high = data[char] > data[char].quantile(0.7)
        med = 1 - low - high

        # form six portfolios, first multiply by market caps
        high_big = mve * (high & big) 
        high_small = mve * (high & small)
        low_big = mve * (low & big)
        low_small = mve * (low & small)
        med_big  = mve * (med & big)
        med_small = mve * (med & small)

        # then divide by sum of market caps in each portfolio
        for ser in [high_big, high_small, low_big, low_small, med_big, med_small]:
            if ser.sum() != 0:
                ser /= ser.sum()
        
        # calculate long-short portfolio for each factor
        factor = 0.5 * (
            high_big + high_small  
            - low_big - low_small
        )

        # add to dictionary
        factor_dct[name_dct[char]] = factor.to_numpy()

        # define SMB in original Fama-French way, using only book-to-market
        if char == "bm":   
            smb = (
                high_small + med_small + low_small 
                - high_big - med_big - low_big 
            ) / 3  
            factor_dct["smb"] = smb.to_numpy() 

    df = pd.DataFrame(factor_dct)
    df.index = data.index

    # CMA is low minus high
    if "cma" in names:
        df["cma"] *= -1

    # VW market port
    df['mkt_rf'] = mve / mve.sum()
    
    return df.to_numpy()

# calculate fama-macbeth weights for a single month
def fama_macbeth(data, chars, **kwargs):
    d = data.dropna()
  
    # standardize characteristics
    #d = d.apply(
    #    lambda x: x / x.std() if x.std() != 0 else 0, 
    #    axis=0
    #) 

    # FM portfolios
    X = d.to_numpy()
    X = np.concatenate((np.ones(X.shape[0]).reshape(-1, 1), X), axis=1)
    P = X @ linalg.pinvh(X.T @ X) 
    P = pd.DataFrame(P[:, 1:])
    P *= 2 / P.abs().sum()
    if len(chars) == 5:
        P.columns = ["smb", "hml", "cma", "rmw", "umd"]
    else:
        P.columns = ["smb", "hml", "umd"]

    # EW market port
    P['mkt_rf'] = 1 / len(data)
    
    return P.to_numpy()

def all_stocks(data, **kwargs):
    return np.eye(len(data.dropna()))

# panel of factor returns for method in (fama_french, fama_macbeth, all_stocks)
def factors(method, panel, n_jobs, start, end, chars):
    def monthly_rets(month):
        data = panel.loc[month]
        weights = method(data[chars], chars, mve=data.mve)
        rets = data.xret.to_numpy().reshape(-1, 1)
        wts = weights.T @ rets
        wts = pd.DataFrame(wts.T, index=[month])
        return wts
    lst = Parallel(n_jobs=n_jobs, verbose=0)(
       delayed(monthly_rets)(month) for month in range(start, end+1)
    )
    #lst = [monthly_rets(month) for month in range(start, end+1)]

    # stacking row vectors on top of each other produces a series of row vectors
    # stacking column vectors next to each other produces a dataframe w multiple rows
    # so stack column vectors next to each other, then transpose
    f = pd.concat(lst)
    f.index.name = "month"
    return f

# portfolio of factors from past factor returns 
def mve_data(f, month, alpha):
    X = f.loc[month-360:month-1].dropna().to_numpy() 
    y = np.ones(len(X))
    ridge = Ridge(fit_intercept=False, alpha=360*alpha)
    pi = ridge.fit(X=X, y=y).coef_
    return pd.Series(pi, index=f.columns)
