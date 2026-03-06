import numpy as np
import matplotlib.pyplot as plt


def aic(N, RSS, k):
    """
    Compute Akaike Information Criterion (AIC)
    """

    return N * np.log(RSS / N) + 2 * k


def bic(N, RSS, k):
    """
    Compute Bayesian Information Criterion (BIC)
    """

    return N * np.log(RSS / N) + k * np.log(N)


def plot_fit_2d(x, y, z, model_func, beta, degree):
    """
    Plot 2D polynomial regression surface and data.

    Parameters
    ----------
    x : array
        x spatial coordinates
    y : array
        y spatial coordinates
    z : array
        observed values (ln I)
    model_func : function
        fitted model
    beta : array
        estimated parameters
    """

    grid_size = 100

    x_lin = np.linspace(min(x), max(x), grid_size)
    y_lin = np.linspace(min(y), max(y), grid_size)

    X, Y = np.meshgrid(x_lin, y_lin)

    Z_pred = model_func(X, Y, beta)

    fig = plt.figure(figsize=(12,4.5))

    ax1 = fig.add_subplot(121, projection='3d', alpha=0.5)

    ax1.scatter(x, y, z, alpha=0.08, color='blue')
    ax1.set_title("Observed Data")
    ax1.set_xlabel("x (nm)")
    ax1.set_ylabel("y (nm)")
    ax1.set_zlabel("ln(I)")


    ax2 = fig.add_subplot(122, projection='3d')

    ax2.plot_surface(X, Y, Z_pred, cmap='viridis', alpha=0.8)
    ax2.set_title("Fitted Surface")
    ax2.set_xlabel("x (nm)")
    ax2.set_ylabel("y (nm)")
    ax2.set_zlabel("ln(I)")

    plt.tight_layout()
    plt.show()

def plot_current_maps(x, y, z, beta, degree, scan_size_nm=22.5):

    # reconstruct grid size
    N = int(np.sqrt(len(z)))

    # reshape original map
    Z = z.reshape(N, N)

    # spatial grid
    x_grid = np.linspace(0, scan_size_nm, N)
    y_grid = np.linspace(0, scan_size_nm, N)

    X, Y = np.meshgrid(x_grid, y_grid)

    # predicted log current
    z_pred = np.zeros_like(X)

    idx = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            z_pred += beta[idx] * (X**i) * (Y**j)
            idx += 1

    # convert back to current
    I_pred = np.exp(z_pred)

    # plotting
    fig, ax = plt.subplots(1, 2, figsize=(12,5))

    im1 = ax[0].imshow(Z,
                       extent=[0,scan_size_nm,0,scan_size_nm],
                       origin="lower",
                       cmap="viridis")

    ax[0].set_title("Original Current Map")
    ax[0].set_xlabel("x (nm)")
    ax[0].set_ylabel("y (nm)")
    plt.colorbar(im1, ax=ax[0])

    im2 = ax[1].imshow(I_pred,
                       extent=[0,scan_size_nm,0,scan_size_nm],
                       origin="lower",
                       cmap="viridis")

    ax[1].set_title("Predicted Current Map")
    ax[1].set_xlabel("x (nm)")
    ax[1].set_ylabel("y (nm)")
    plt.colorbar(im2, ax=ax[1])

    plt.tight_layout()
    plt.show()

def plot_projection(x, y, z, model_func, beta, degree):

    z_pred = model_func(x, y, beta)

    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].scatter(x, z, alpha=0.06, label="Observed", color='blue')
    ax[0].plot(x, z_pred, alpha=0.4, label="Predicted", color='red')
    ax[0].set_title("Projection on x-axis")
    ax[0].set_xlabel("x (nm)")
    ax[0].set_ylabel("ln(I)")
    ax[0].legend()
    ax[1].scatter(y, z, alpha=0.06, label="Observed", color='blue')
    ax[1].plot(y, z_pred, alpha=0.4, label="Predicted", color='red')
    ax[1].set_title("Projection on y-axis")
    ax[1].set_xlabel("y (nm)")
    ax[1].set_ylabel("ln(I)")
    ax[1].legend()
    plt.tight_layout()
    plt.show()


    