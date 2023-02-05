import sys
sys.path.insert('/home/michaelnaps/prog/ode');

import numpy as np
import matplotlib.pyplot as plt

import ode
import Helpers.KoopmanFunctions as kman


# create global "measurement" variables
Nu = 2;
Na = 4;
anchors = np.random.rand(2,Na);


def obsX(x=None):
    if x is None:
        meta = {'Nk':4};
        return meta;
    return x;

def obsU(x=None):
    if x is None:
        meta = {'Nk':1};
        return meta;
    return 1;

def obsH(X=None):
    if X is None:
        meta = {'Nk':Na+Nu};
        return meta;

    x = X[:4];
    u = X[4:];

    xp = x[:2].T[0];

    dist = np.zeros( (Na,1) );
    for i, anchor in enumerate(anchors.T):
        dist[i] = (anchor.T - xp).T @ (anchor.T - xp);

    Psi = np.vstack( (dist, u) );

    return Psi;

def obs(X=None):
    if X is None:
        meta = {'Nk': obsX()['Nk'] + obsU()['Nk']*2};
        return meta;

    x = X[:4];
    u = X[4:];

    PsiX = obsX(x);
    PsiU = obsU(x);
    PsiH = u;

    Psi = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );

    return Psi;

def plot(T, X, U):
    fig, axs = plt.subplots(2,1);
    axs[0].scatter(X[0], X[1]);
    axs[1].plot(T, U[0]);
    axs[1].plot(T, U[1]);
    return fig, axs;


# set global output setting
np.set_printoptions(precision=3, suppress=True);


if __name__ == "__main__":
    # construct model
    Nx = 4;

    dt = 0.1;
    xg = np.zeros( (Nx,1) );
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
    C = [
        [10, 0, 2.5, 0],
        [0, 10, 0, 2.5]
    ];

    model = lambda x,u: A@x + B@u;
    control = lambda x: C@(xg.reshape(Nx,1) - x.reshape(Nx,1));


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

    plot(tTrain[:Nt-1], xTrain, uRand);
    # plt.show()


    # construct Kx data matrices
    X = np.vstack( (xTrain[:,:Nt-1], uRand) );
    Y = np.vstack( (xTrain[:,1:Nt], uRand) );

    X0 = np.vstack( (x0, uRand[:,0].reshape(Nu,1)) );

    kxvar = kman.KoopmanOperator(obs);
    Kx = kxvar.edmd(X, Y, X0);

    print('Kx\n', Kx);


    # construct data for Ku
    xRand = np.random.rand(Nx,Nt-1);
    uTrain = np.zeros( (Nu,Nt-1) );

    for i in range(Nt-1):
        uTrain[:,i] = control(xRand[:,i]).reshape(Nu,);

    plot(tTrain[:Nt-1], xRand, uTrain);
    # plt.show()


    # solve for Ku
    Xu = np.vstack( (xRand, np.zeros( (Nu,Nt-1) )) );
    Yu = np.vstack( (xRand, uTrain) );

    Xu0 = np.vstack( (
        xRand[:,0].reshape(Nx,1), uTrain[:,0].reshape(Nu,1)
    ) );
    kuvar = kman.KoopmanOperator(obsH);
    Ku = kuvar.edmd(Xu, Yu, Xu0)

    print('Ku\n', Ku);


    # calculate cumulative operator
    m = Nu;
    p = obsX()['Nk'];
    q = obsU()['Nk'];
    b = obsH()['Nk'];

    K = Kx @ np.vstack( (
        np.hstack( (np.eye(p), np.zeros( (p,q*b) )) ),
        np.hstack( (np.zeros( (m*q, p) ), np.kron(np.eye(q), Ku[4:,:])) )
    ) );

    print('K\n', K)
