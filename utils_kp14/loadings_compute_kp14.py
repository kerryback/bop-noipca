import numpy as np
import pandas as pd
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

# function to compute loadings based on Taylor
def loadings_Taylor(K, x, z, eps, uj, rate, high, lambda_f, P, G_up, G_down):
    Et_x = np.exp(mu_x*dt)*x
    alph = alpha/(1 - alpha)
    Et_z = z*np.exp(mu_z*dt)
    Et_eps = 1+ (eps -1)*np.exp(-theta_eps*dt)
    Et_uj = 1 + (uj - 1)*np.exp(-theta_u*dt)
    EtA = A(1+ (eps[np.newaxis, :, :] -1)*np.exp(-theta_eps*dt), Et_uj)
    G_approx = lambda_f*(G_down(Et_eps) + 
                        (high*(1 - mu_H*dt) + (1-high)*mu_L*dt)*(G_up(Et_eps) - G_down(Et_eps)))
    
    # compute A_z
    term1 = alph*rate*dt*C*Et_z**((2*alpha-1)/(1 - alpha))*Et_x*A(Et_eps, 1)**(1/(1 - alpha))
    term2 = alph*Et_z**((2*alpha-1)/(1 - alpha))*Et_x*G_approx
    A_z = (term1 + term2)/P

    # compute A_x
    term1 = (1 - delta*dt)*np.sum(EtA*K**alpha, axis = 0)
    term2 = rate*dt*C*Et_z**alph*A(Et_eps, 1)**(1/(1 - alpha))
    term3 = Et_z**alph*G_approx
    A_x = (term1 + term2 + term3)/P

    return A_z, A_x

# function to compute loadings based on projection
def loadings_projection(K, x, z, eps, uj, rate, high, lambda_f, P, eret, Et_G_up, Et_G_down, Et_A_mod):
    Et_z = np.exp(mu_z*dt)*z
    Et_x = np.exp(mu_x*dt)*x
    Vart_z = z**2*np.exp(2*mu_z*dt)*(np.exp(sigma_z**2*dt) - 1)
    Vart_x = x**2*np.exp(2*mu_x*dt)*(np.exp(sigma_x**2*dt) - 1)
    alph = alpha/(1 - alpha)
    Et_z_alph = z**alph*np.exp(alph*mu_z*dt + 0.5*alph*(2*alpha-1)/(1 - alpha)*sigma_z**2*dt)
    alph2 = 1/(1 - alpha)
    Et_z_alph2 = z**alph2*np.exp(alph2*mu_z*dt + 0.5*alph2*(alph2 -1)*sigma_z**2*dt)
    Et_x2 = x**2*np.exp(2*mu_x*dt + sigma_x**2*dt)

    EtA = A(1+ (eps[np.newaxis, :, :] -1)*np.exp(-theta_eps*dt), 1 + (uj - 1)*np.exp(-theta_u*dt))
    Et_G = ((high == 0)*lambda_f*((1 - mu_L*dt)* Et_G_down(eps) + mu_L*dt* Et_G_up(eps)) + 
        (high == 1)*lambda_f*((1 - mu_H*dt)* Et_G_up(eps) + mu_H*dt * Et_G_down(eps)))
    rate = lambda_f*(lambda_L + (lambda_H - lambda_L)*high)

    term1 = (1 - delta*dt)*np.sum(EtA*K**alpha, axis = 0)
    term2 = Et_G
    term3 = np.sum(eps[np.newaxis, :, :]*uj*x[np.newaxis, :]*K**alpha*dt, axis = 0)
    term4 = rate*dt*C*Et_A_mod(eps)

    A_z = (Et_z*Et_x*term1 + Et_z_alph2*Et_x*(term2 + term4) + Et_z*term3 )/P - (1+eret)*Et_z
    A_x = (Et_x2*term1 + Et_z_alph*Et_x2*(term2+term4) + Et_x*term3 )/P - (1+eret)*Et_x
    A_z /= Vart_z
    A_x /= Vart_x

    return A_z, A_x