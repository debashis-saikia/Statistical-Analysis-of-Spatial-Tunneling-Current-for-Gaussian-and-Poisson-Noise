import numpy as np
import pandas as pd
from scipy.optimize import minimize
from GaussianNoise2D import (residual_sum_squares_2d, estimate_sigma2_2d, chi_square_2d, reduced_chi_square_2d)
from model_metrics2d import aic, bic

class PolynomialRegression2D:

    def __init__(self, degree):
        self.degree = degree
        self.beta_hat = None
        self.RSS = None
        self.sigma2_hat = None
        self.sigma_hat = None
        self.chi2_red = None
        self.AIC = None
        self.BIC = None

    def polynomial_model(self, x, y, beta):

        z_pred = np.zeros_like(x)

        idx = 0
        for i in range(self.degree + 1):
            for j in range(self.degree + 1 - i):
                z_pred += beta[idx] * (x**i) * (y**j)
                idx += 1

        return z_pred

    def objective(self, beta, x, y, z):

        return residual_sum_squares_2d(
            x,
            y,
            z,
            beta,
            lambda x, y, b: self.polynomial_model(x, y, b)
        )

    def fit(self, x, y, z):

        self.N = len(z)

        k = (self.degree + 1) * (self.degree + 2) // 2
        beta_init = np.ones(k)

        result = minimize(
            self.objective,
            beta_init,
            args=(x, y, z)
        )

        self.beta_hat = result.x

        self.RSS = residual_sum_squares_2d(
            x,
            y,
            z,
            self.beta_hat,
            lambda x, y, b: self.polynomial_model(x, y, b)
        )

        self.sigma2_hat = estimate_sigma2_2d(
            x,
            y,
            z,
            self.beta_hat,
            lambda x, y, b: self.polynomial_model(x, y, b)
        )

        self.sigma_hat = np.sqrt(self.sigma2_hat)

        self.chi2_red = reduced_chi_square_2d(
            x,
            y,
            z,
            self.beta_hat,
            self.sigma_hat,
            lambda x, y, b: self.polynomial_model(x, y, b),
            k - 1
        )

        self.AIC = aic(self.N, self.RSS, k)
        self.BIC = bic(self.N, self.RSS, k)

    def predict(self, x, y):
        return self.polynomial_model(x, y, self.beta_hat)

    def model_info(self):
        print("Polynomial Regression 2D Model Information")
        print(f"Polynomial degree: {self.degree}")
        print(f"Estimated parameters (beta): {self.beta_hat}")
        print(f"RSS = {self.RSS}")
        print(f"Estimated sigma^2 = {self.sigma2_hat}")
        print(f"Reduced Chi-square = {self.chi2_red}")
        print(f"AIC = {self.AIC}")
        print(f"BIC = {self.BIC}"   )