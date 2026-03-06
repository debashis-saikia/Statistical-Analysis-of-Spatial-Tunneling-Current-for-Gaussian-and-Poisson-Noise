import numpy as np

def gaussian_model(x, beta, model_func, sigma):
    """
    Generates observations using Gaussian noise model.

    Parameters
    ----------
    x : array
        Input data
    beta : array
        Model parameters
    model_func : function
        Function f(x, beta)
    sigma : float
        Standard deviation of noise

    Returns
    -------
    y : array
        Noisy observations
    """
    
    epsilon = np.random.normal(0, sigma, size=len(x))
    y = model_func(x, beta) + epsilon
    
    return y

def gaussian_likelihood(x, y, beta, sigma, model_func):
    """
    Computes likelihood of Gaussian noise model.
    """

    residuals = y - model_func(x, beta)
    
    N = len(y)
    
    coeff = 1 / np.sqrt(2 * np.pi * sigma**2)
    
    likelihood = np.prod(
        coeff * np.exp(-(residuals**2) / (2 * sigma**2))
    )
    
    return likelihood

def log_gaussian_likelihood(x, y, beta, sigma, model_func):
    """
    Computes log-likelihood of Gaussian model.
    """
    
    residuals = y - model_func(x, beta)
    N = len(y)

    logL = (
        - (N/2) * np.log(2*np.pi*sigma**2)
        - (1/(2*sigma**2)) * np.sum(residuals**2)
    )
    
    return logL

def residual_sum_squares(x, y, beta, model_func):
    """
    Computes RSS.
    """
    
    residuals = y - model_func(x, beta)
    
    RSS = np.sum(residuals**2)
    
    return RSS

def estimate_sigma2(x, y, beta, model_func):
    """
    Estimate noise variance.
    """
    
    RSS = residual_sum_squares(x, y, beta, model_func)
    
    N = len(y)
    
    sigma2 = RSS / N
    
    return sigma2

def chi_square(x, y, beta, sigma, model_func):
    """
    Computes chi-square statistic.
    """
    
    y_hat = model_func(x, beta)
    
    chi2 = np.sum((y - y_hat)**2 / sigma**2)
    
    return chi2

def degrees_of_freedom(N, p):
    """
    Compute degrees of freedom.
    
    N : number of data points
    p : number of parameters
    """
    
    return N - (p + 1)

def reduced_chi_square(x, y, beta, sigma, model_func, p):
    """
    Computes reduced chi-square.
    """
    
    chi2 = chi_square(x, y, beta, sigma, model_func)
    
    nu = degrees_of_freedom(len(y), p)
    
    return chi2 / nu


