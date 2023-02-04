import numpy as np
import matplotlib.pyplot as plt

import Helpers.KoopmanFunctions as kman


def obsX(x):
    return x;

def obsU(x):
    return 1;

def obs(X=None):
    if X is None:
        meta = {'Nk': 6};
        return meta;

    x = X[:4];
    u = X[4:];

    PsiX = obsX(x);
    PsiU = obsU(x);
    PsiH = u;

    Psi = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );

    return Psi;


# set global output setting
np.set_printoptions(precision=3, suppress=True);


if __name__ == "__main__":
    # construct model
    Nx = 4;
    Nu = 2;

    dt = 0.1;
    A = [
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ];
    B = [
        [0, 0],
        [0, 0],
        [dt, 0],
        [0, dt]
    ];

    model = lambda x,u: A@x + B@u;

    # simulation variables
    T = 10;  Nt = round(T/dt)+1;
    tTrain = np.array( [i*dt for i in range(Nt)] );

    # generate list of randomly assorted u
    x0 = 10*np.random.rand(4,1) - 5;

    uRand = 2*np.random.rand(Nu, Nt-1) - 1;
    xTrain = np.zeros( (Nx,Nt) );

    xTrain[:,1] = x0.reshape(Nx,);

    for i in range(Nt-1):
        xTrain[:,i+1] = model(xTrain[:,i], uRand[:,i]);

    # plot the trajectories
    fig, axs = plt.subplots(2,1);
    axs[0].plot(xTrain[0], xTrain[1]);
    axs[1].plot(tTrain[:Nt-1], uRand[0]);
    axs[1].plot(tTrain[:Nt-1], uRand[1]);
    # plt.show();

    # construct Kx data matrices
    X = np.vstack( (xTrain[:,:Nt-1], uRand) );
    Y = np.vstack( (xTrain[:,1:Nt], uRand) );

    X0 = np.vstack( (x0, uRand[:,0].reshape(Nu,1)) );

    kvar = kman.KoopmanOperator(obs);
    Kx = kvar.edmd(X, Y, X0);

    print(Kx);
