import numpy as np
import matplotlib.pyplot as plt

from Helpers.KoopmanFunctions import *
from KMAN.DataFunctions import *


def obsX(x=None):
    if x is None:
        meta = {'Nk': 2};
        return meta;

    PsiX = x;
    return PsiX;

def obsU(x=None):
    if x is None:
        meta = {'Nk': 1};
        return meta;

    PsiU = [[1]];
    return PsiU;

def obsXU(X=None):
    if X is None:
        meta = {'Nk': obsX()['Nk'] + obsU()['Nk']*obsH()['Nk']};
        return meta;

    x = X[0:2].reshape(2,1);
    u = X[2];

    PsiX = obsX(x);
    PsiU = obsU(x);
    PsiH = obsH(X);

    PsiXU = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );

    return PsiXU;

def obsH(X=None):
    if X is None:
        meta = {'Nk': 1};
        return meta;

    PsiH = X[2];
    return PsiH;


def plot(tlist, X, PSI=None):
    if PSI is not None:
        nrows = 2;
    else:
        nrows = 1;

    fig, ax = plt.subplots(1,nrows);

    if PSI is not None:
        ax[0].plot(tlist, X[0]);
        ax[0].plot(tlist, X[1]);
        ax[0].set_title("Model");

        ax[1].plot(tlist, PSI[0]);
        ax[1].plot(tlist, PSI[1]);
        ax[1].set_title("KCE");
    else:
        ax.plot(tlist, X[0]);
        ax.plot(tlist, X[1]);
        ax.set_title("Model");

    return fig, ax;


if __name__ == "__main__":
    Nx = 2;
    Nu = 1;
    x0 = np.random.rand(Nx,1);
    u0 = np.array( [[0]] );

    # model equations
    xg = np.array( [[0],[0]] );
    dt = 0.1;
    A = np.array( [[1, dt], [0, 1]] );
    B = np.array( [[0], [dt]] );
    C = np.array( [[10, 5]] );

    model = lambda x,u: A @ x.reshape(Nx,1) + B @ u.reshape(Nu,1);

    control = lambda x: C @ (xg.reshape(Nx,1) - x.reshape(Nx,1));
    random_control = lambda x: 10*np.random.rand(Nu,1) - 5;


    # simulate model and control
    T = 5;
    Nt = round(T/dt) + 1;
    tlist = np.array([i*dt for i in range(Nt)]);

    ulist = np.zeros( (Nu, Nt) );
    xlist = np.zeros( (Nx, Nt) );

    ulist[:,0] = u0.reshape(Nu,);
    xlist[:,0] = x0.reshape(Nx,);

    for i in range(Nt-1):
        uNew = control(xlist[:,i]);
        ulist[:,i+1] = uNew.reshape(Nu,);

        xNew = model(xlist[:,i], ulist[:,i+1]);
        xlist[:,i+1] = xNew.reshape(Nx,);

    plot(tlist, xlist);
    plt.show();


    # generate Kx from data
    xu0 = np.vstack( (x0, u0) );
    Xx = np.vstack( (xlist[:,:Nt-1], ulist[:,1:Nt]) );
    Yx = np.vstack( (xlist[:,1:Nt], ulist[:,1:Nt]) );

    Kx_var = KoopmanOperator(obsXU);
    Kx = Kx_var.edmd(Xx, Yx, xu0);

    print('Kx\n', Kx, '\n');
