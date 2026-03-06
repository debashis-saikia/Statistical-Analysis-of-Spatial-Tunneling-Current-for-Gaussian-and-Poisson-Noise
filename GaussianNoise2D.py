import numpy as np


def gaussian_model_2d(x, y, beta, model_func, sigma):
    """
    Generates observations using Gaussian noise model for 2D spatial data.

    Parameters
    ----------
    x : array
        x coordinates
    y : array
        y coordinates
    beta : array
        Model parameters
    model_func : function
        Function f(x, y, beta)
    sigma : float
        Standard deviation of noise

    Returns
    -------
    z : array
        Noisy observations
    """

    epsilon = np.random.normal(0, sigma, size=len(x))
    z = model_func(x, y, beta) + epsilon

    return z


def gaussian_likelihood_2d(x, y, z, beta, sigma, model_func):
    """
    Computes likelihood of Gaussian noise model for 2D fitting.
    """

    residuals = z - model_func(x, y, beta)

    N = len(z)

    coeff = 1 / np.sqrt(2 * np.pi * sigma**2)

    likelihood = np.prod(
        coeff * np.exp(-(residuals**2) / (2 * sigma**2))
    )

    return likelihood


def log_gaussian_likelihood_2d(x, y, z, beta, sigma, model_func):
    """
    Computes log-likelihood of Gaussian model for 2D data.
    """

    residuals = z - model_func(x, y, beta)
    N = len(z)

    logL = (
        - (N/2) * np.log(2*np.pi*sigma**2)
        - (1/(2*sigma**2)) * np.sum(residuals**2)
    )

    return logL


def residual_sum_squares_2d(x, y, z, beta, model_func):
    """
    Computes RSS for 2D fitting.
    """

    residuals = z - model_func(x, y, beta)

    RSS = np.sum(residuals**2)

    return RSS


def estimate_sigma2_2d(x, y, z, beta, model_func):
    """
    Estimate noise variance for 2D model.
    """

    RSS = residual_sum_squares_2d(x, y, z, beta, model_func)

    N = len(z)

    sigma2 = RSS / N

    return sigma2


def chi_square_2d(x, y, z, beta, sigma, model_func):
    """
    Computes chi-square statistic for 2D data.
    """

    z_hat = model_func(x, y, beta)

    chi2 = np.sum((z - z_hat)**2 / sigma**2)

    return chi2


def degrees_of_freedom(N, p):
    """
    Compute degrees of freedom.

    N : number of data points
    p : number of parameters
    """

    return N - (p + 1)


def reduced_chi_square_2d(x, y, z, beta, sigma, model_func, p):
    """
    Computes reduced chi-square for 2D model.
    """

    chi2 = chi_square_2d(x, y, z, beta, sigma, model_func)

    nu = degrees_of_freedom(len(z), p)

    return chi2 / nu