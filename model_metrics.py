import numpy as np
import matplotlib.pyplot as plt

def aic(N, RSS, k):
    """
    Compute Akaike Information Criterion (AIC)

    Parameters
    ----------
    N : int
        Number of data points
    RSS : float
        Residual Sum of Squares
    k : int
        Number of model parameters

    Returns
    -------
    float
        AIC value
    """

    return N * np.log(RSS / N) + 2 * k


def bic(N, RSS, k):
    """
    Compute Bayesian Information Criterion (BIC)

    Parameters
    ----------
    N : int
        Number of data points
    RSS : float
        Residual Sum of Squares
    k : int
        Number of model parameters

    Returns
    -------
    float
        BIC value
    """

    return N * np.log(RSS / N) + k * np.log(N)

def plot_fit(x, y, model_func, beta):
    """
    Plot data and model fit.
    """
    
    x_fit = np.linspace(min(x), max(x), 500)
    y_fit = model_func(x_fit, beta)

    plt.scatter(x, y, label="Data", alpha=0.5, color='blue')
    plt.plot(x_fit, y_fit, label="Polynomial Fit", alpha=0.8, color='red')
    plt.xlabel("Distance (nm)")
    plt.ylabel("ln(Current)")
    plt.legend()
    plt.show()
