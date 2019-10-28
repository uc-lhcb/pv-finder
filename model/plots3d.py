import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

__all__ = ("cube_marginals", "plotcube")


def cube_marginals(cube, *, normalize=False):

    c_fcn = np.mean if normalize else np.sum

    xy = c_fcn(cube, axis=0)
    xz = c_fcn(cube, axis=1)
    yz = c_fcn(cube, axis=2)
    return xy, xz, yz


def plotcube(
    cube, x=None, y=None, z=None, *, normalize=False, plot_front=False, ax=None
):
    """Use contourf to plot cube marginals"""

    Z, Y, X = cube.shape
    xy, xz, yz = cube_marginals(cube, normalize=normalize)

    if x == None:
        x = np.arange(X)
    if y == None:
        y = np.arange(Y)
    if z == None:
        z = np.arange(Z)

    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection="3d")
    else:
        fig = ax.parent

    ax.plot_surface(
        x[None, :].repeat(Y, axis=0),
        y[:, None].repeat(X, axis=1),
        xy,
        cmap=plt.cm.coolwarm,
        alpha=0.75,
    )

    ax.plot_surface(
        x[None, :].repeat(Z, axis=0),
        xz,
        z[:, None].repeat(X, axis=1),
        cmap=plt.cm.coolwarm,
        alpha=0.75,
    )

    ax.plot_surface(
        yz,
        y[None, :].repeat(Z, axis=0),
        z[:, None].repeat(Y, axis=1),
        cmap=plt.cm.coolwarm,
        alpha=0.75,
    )

    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
