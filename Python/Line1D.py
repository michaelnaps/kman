from KoopmanSolve import *

import numpy as np
import matplotlib.pyplot as plt

class MetaVariable:
    def __init__(self, Nk):
        self.Nk = Nk;
        return;

def obsX(x):
    Nx = 4;
    Nk = 5;

    PsiX = np.zeros((Nk,1));
    PsiX[0:Nx,:] = x;
    PsiX[Nx,:] = [1];

    metaX = MetaVariable(Nk);
    metaX.x = [1,2,3,4];
    metaX.c = [5];

    return (PsiX, metaX);

def obsU(x):
    Nk = 5;
    (PsiU, _) = obsX(x);

    metaU = MetaVariable(Nk);
    metaU.xu = [1,2,3,4];
    metaU.cu = [5];

    return (PsiU, metaU);

def obsXU(x, u):
    Nx = 4
    Nu = 2;

    (PsiX, metaX) = obsX(x);
    (PsiU, metaU) = obsU(x);
    Nk = metaX.Nk + Nu*metaU.Nk;

    PsiXU = np.zeros((Nk,1));
    PsiXU[0:metaX.Nk,:] = PsiX;
    PsiXU[metaX.Nk:Nk,:] = np.kron(PsiU, u);

    metaXU = MetaVariable(Nk);
    return (PsiXU, metaXU);

def obsH(X):
    Nx = 4;
    Nu = 2;
    Nk = Nx + Nu + 1;
    metaH = MetaVariable(Nk);

    x = X[:Nx,:];
    u = X[Nx:,:];

    PsiH = np.vstack( (x, u, [1]) );

    return (PsiH, metaH);

def plot(tlist, xlist, ulist):
    fig, ax = plt.subplots(2,1);

    ax[0].plot(tlist, xlist[0,:]);
    ax[0].plot(tlist, xlist[1,:]);

    ax[1].plot(tlist, ulist[0,:]);
    ax[1].plot(tlist, ulist[1,:]);

    return fig, ax;

if __name__ == "__main__":
    # establish model equations and controller
    dt = 0.1;
    g = np.array( [[1],[-1],[0],[0]] );
    c = 0.25;

    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1-c, 0],
        [0, 0, 0, 1-c]
    ]);
    B = np.array([
        [0, 0],
        [0, 0],
        [dt, 0],
        [0, dt]
    ]);
    C = np.array([
        [10, 0, 2.5, 0],
        [0, 10, 0, 2.5]
    ]);

    model = lambda x,u: A@x.reshape(Nx,1) + B@u.reshape(Nu,1);
    control = lambda x: C@(g.reshape(Nx,1) - x.reshape(Nx,1));


    # establish initial conditions
    Nx = 4;
    Nu = 2;
    x0 = np.array( [[0],[0],[0],[0]] );
    u0 = np.array( [[0],[0]] );


    # simulate system model
    T = 5;
    Nt = round(T/dt)+1;

    tlist = np.array( [[i*dt] for i in range(Nt)] );
    xlist = np.zeros( (Nx,Nt) );
    ulist = np.zeros( (Nu,Nt) );

    for i in range(Nt-1):
        uNew = control(xlist[:,i]);
        ulist[:,i+1] = uNew.reshape(Nu,)

        xNew = model(xlist[:,i], ulist[:,i+1]);
        xlist[:,i+1] = xNew.reshape(Nx,)

    plot(tlist, xlist, ulist);


    # get observable function meta-data
    (_, metaH) = obsH(np.vstack( (x0, u0) ));
    (_, metaX) = obsX(x0);
    (_, metaU) = obsU(x0);
    (_, metaXU) = obsXU(x0, u0);


    # generate Ku
    Xu = np.vstack( (xlist, np.zeros( (Nu,Nt) )) );
    Yu = np.vstack( (xlist, ulist) );

    (Ku, err, ind) = KoopmanSolve(obsH, metaH.Nk, Xu, Yu, np.vstack( (x0,u0) ));
