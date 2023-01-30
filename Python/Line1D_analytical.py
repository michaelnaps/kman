import numpy as np
import matplotlib.pyplot as plt

from Helpers.KoopmanFunctions import *
from Helpers.DataFunctions import *


def obsX(x=None):
    if x is None:
        Nk = 2;
        return Nk;

    PsiX = x;
    return PsiX;

def obsU(x=None):
    if x is None:
        Nk = 1;
        return Nk;

    PsiU = [[1]];
    return PsiU;

def obsXU(X=None):
    if X is None:
        Nk = obsX() + obsU()*obsH();
        return Nk;

    x = X[0:2].reshape(2,1);
    u = X[2];

    PsiX = obsX(x);
    PsiU = obsU(x);
    PsiH = obsH(X);

    PsiXU = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );

    return PsiXU;

def obsH(X=None):
    if X is None:
        Nk = 3;
        return Nk;

    # switch order
    x = X[:2].reshape(2,1);
    u = X[2];

    PsiH = np.vstack( (u, x) );
    return PsiH;


def plot(tlist, X, PSI):
    fig, ax = plt.subplots(1,2);
    ax[0].plot(tlist, X[0]);
    ax[0].plot(tlist, X[1]);
    ax[0].set_title("Model");

    ax[1].plot(tlist, PSI[0]);
    ax[1].plot(tlist, PSI[1]);
    ax[1].set_title("KCE");

    return fig, ax;


if __name__ == "__main__":
    Nx = 2;
    Nu = 1;
    x0 = np.array( [[1],[1]] );
    u0 = np.array( [[0]] );

    # model equations
    xg = np.array( [[0],[0]] );
    dt = 0.1;
    A = np.array( [[1, dt], [0, 1]] );
    B = np.array( [[0], [dt]] );
    C = np.array( [[10, 5]] );

    model = lambda x,u: A @ x.reshape(Nx,1) + B @ u.reshape(Nu,1);
    control = lambda x: C @ (xg.reshape(Nx,1) - x.reshape(Nx,1));


    # simulate model and control (for comparisons)
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


    # analytically construct Ku
    NkH = obsH();
    Ku = np.vstack( (
        np.hstack( ([[0]], -C) ),
        np.hstack( ([[0],[0]], np.eye(2)) )
    ) );

    print('Ku\n', Ku, '\n');

    # intialize and construct Kx
    NkXU = obsXU();

    Kx = np.vstack( (
        np.hstack( (A, B, [[0,0], [0,0]]) ),
        np.hstack( (np.zeros( (3,2) ), np.eye( 3 )) )
    ) );

    print('Kx\n', Kx, '\n');


    # generate the cumulative operator
    # K = Kx*[I in (p x p), 0 in (p x mq); 0 in (mq x p), kron(Ku, I in q)]
    m = Nu;
    p = obsX();
    q = obsU();
    b = NkH;

    top = np.hstack( (np.eye(p), np.zeros( (p, b*q) )) );
    bot = np.hstack( (np.zeros( (b*q, p) ), np.kron(Ku, np.eye(q))) );

    K = Kx @ np.vstack( (top, bot) );

    print('K\n', K, '\n')


    # test input-specific operator
    xmodel = np.zeros( (Nx, Nt) );
    xmodel[:,0] = x0.reshape(Nx,);

    psitest = np.zeros( (NkH, Nt) );
    psitest[:,0] = obsH(np.vstack( (x0, u0) )).reshape(NkH,);

    for i in range(Nt-1):
        Psi_c = obsH(np.vstack((xmodel[:,i].reshape(Nx,1), u0)));
        Psi_n = Ku @ Psi_c;

        psitest[:,i+1] = Psi_n.reshape(NkH,)
        xmodel[:,i+1] = model(xmodel[:,i], Psi_n[2]).reshape(Nx,);

    plot(tlist, xlist, psitest)
    plt.show()


    # test cumulative operator
    psilist = np.zeros( (NkXU, Nt) );
    psilist[:,0] = obsXU( np.vstack( (x0,u0) ) ).reshape(NkXU,);

    for i in range(Nt-1):
        Psi_n = K @ psilist[:,i].reshape(NkXU,1);
        psilist[:,i+1] = Psi_n.reshape(NkXU,);

    plot(tlist, xlist, psilist);
    plt.show();
