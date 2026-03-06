import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from model_metrics import aic, bic, plot_fit
from GaussianNoise import (residual_sum_squares, estimate_sigma2, chi_square, reduced_chi_square)

data = pd.read_csv(r"C:\Users\IISER13\OneDrive\Desktop\PHY 5032\processed_tunneling_dataset_1.csv")

x = data["distance_nm"].values
y = data["log_current"].values

N = len(y)

def polynomial_model(x, beta):
    """
    Polynomial regression model.
    beta = [b0, b1, b2, ... bp]
    """  
    y_pred = np.zeros_like(x)
    for i in range(len(beta)):
        y_pred += beta[i] * x**i
    return y_pred

def objective(beta, x, y):
    return residual_sum_squares(x, y, beta, polynomial_model)

p = 1
beta_init = np.ones(p + 1)
result = minimize(objective, beta_init, args=(x, y))
beta_hat = result.x
RSS = residual_sum_squares(x, y, beta_hat, polynomial_model)
sigma2_hat = estimate_sigma2(x, y, beta_hat, polynomial_model)
sigma_hat = np.sqrt(sigma2_hat)
chi2_red = reduced_chi_square(x, y, beta_hat, sigma_hat, polynomial_model, p)
N = len(y)
k = p + 1
AIC = aic(N, RSS, k)
BIC = bic(N, RSS, k)

print('-'*50)
print("Model Selection Metrics:")
print('-'*50)
print("Polynomial degree:", p)
print("\nEstimated parameters (beta):")
print(beta_hat)
print("\nRSS =", RSS)
print("\nEstimated sigma^2 =", sigma2_hat)
print("\nReduced Chi-square =", chi2_red)
print(f'\nAIC for degree {p}:', AIC)
print(f'BIC for degree {p}:', BIC)
plot_fit(x, y, polynomial_model, beta_hat)
