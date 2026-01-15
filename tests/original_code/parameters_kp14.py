# Import burnin from main config.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import KP14_BURNIN as burnin
dt = 1/12

mu_x, mu_z, sigma_x, sigma_z = 0.01, 0.005, 0.13, 0.035
theta_eps, sigma_eps = 0.35, 0.2
theta_u, sigma_u = 0.5, 1.5
delta, mu_lambda, sigma_lambda, mu_H, mu_L, lambda_H = 0.1, 2.0, 2.0, 0.075, 0.16, 2.35
lambda_L = (1 - mu_H/(mu_H + mu_L)*lambda_H)/(1 - mu_H/(mu_H + mu_L))
r, gamma_x, gamma_z = 0.05, 0.69, -0.35 ## NOTE: r is different from KP14
alpha = 0.85


prob_H = (mu_L)/(mu_H + mu_L)
const = r + gamma_x * sigma_x + delta - mu_x
A_0 = 1/const
A_1 = 1/(const + theta_eps)
A_2 = 1/(const + theta_u)
A_3 = 1/(const + theta_eps + theta_u)
A = lambda ep, u: (A_0 + (ep - 1) * A_1 + (u - 1) * A_2 + (ep - 1) * (u - 1) * A_3)

rho = r + gamma_x * sigma_x - mu_x - alpha / (1 - alpha) * (mu_z - gamma_z * sigma_z - 0.5 * sigma_z**2) \
      - 0.5 * (alpha / (1 - alpha))**2 * sigma_z**2
C = alpha**(1 / (1 - alpha)) * (alpha**(-1) - 1)