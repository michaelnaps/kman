import sys
sys.path.insert(0, '/home/michaelnaps/prog/ode');

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.path as path

import ode
import Helpers.KoopmanFunctions as kman
import Helpers.DataFunctions as data


# hyper parameter(s)
dt = 0.01;


# set global output setting
np.set_printoptions(precision=3, suppress=True);

def modelFunc(X):
    mu = 1;
    x = X[0];  dx = X[1];

    xn = X.reshape(Nx,1) + dt*np.array( [
        [dx],
        [mu*(dx - dx*x**2) - x]
    ] );

    return xn;


def obsX(x=None):
    if x is None:
        meta = {'Nk':2};
        return meta;
    Psi = x;
    return Psi;

def obsU(x=None):
    if x is None:
        meta = {'Nk':2};
        return meta;
    Psi = np.vstack( ([1], x[1]) );
    return Psi;

def obsH(x=None):
    if x is None:
        meta = {'Nk':1};
        return meta;
    Psi = x[0]**2;
    return Psi;


if __name__ == "__main__":
    # model parameters
    Nx = 2;
    N0 = 5;
    X0 = 10*np.random.rand(Nx,N0) - 5;

    # create model data
    T = 100;
    Nt = round(T/dt) + 1;

    tList = np.array( [[i*dt for i in range(Nt)]] );

    xTrain, _ = data.generate_data(tList, modelFunc, X0);

    # observables for TRAINING
    def obsTrain(x=None):
        if x is None:
            meta = {'Nk': obsX()['Nk'] + obsU()['Nk']*obsH()['Nk']}
            return meta;

        PsiX = obsX(x);
        PsiU = obsU(x);
        PsiH = obsH(x);

        Psi = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );

        return Psi;

    # organize data
    xData = xTrain[:,:Nt-1];
    yData = xTrain[:,1:Nt];

    X = data.stack_data(xData, N0, Nx, Nt-1);
    Y = data.stack_data(yData, N0, Nx, Nt-1);

    kvar = kman.KoopmanOperator(obsTrain);
    K = kvar.edmd(X, Y, X0);
    Kup = K[:Nx,:];

    print('K:', kvar.err);
    print(K, '\n');

    print('Kup:');
    print(Kup, '\n');

    # define observable function FOR IMPLEMENTATION
    def obsImplm(x=None):
        Nk = obsTrain()['Nk'];
        if x is None:
            meta = {'Nk':Nk};
            return meta;

        PsiX = x;
        PsiU = np.vstack( ([1], x[1]) );
        PsiH = obsH(x);

        Psi = np.vstack( (x, np.kron(x, PsiH)) );
        return Psi.reshape(Nk,1);

    # compare koopman to real model
    Nk = obsTrain()['Nk'];
    Psi0 = X0[:,0].reshape(Nx,1); #obsTrain(X0[:,0].reshape(Nx,1));
    xComp = xTrain[:Nx,:];

    koopFunc = lambda x: Kup@obsImplm(x);
    PsiTest, _ = data.generate_data(tList, koopFunc, Psi0);
    xTest = PsiTest[:Nx,:];

    # plot comparison
    fig, axs = plt.subplots();
    axs.plot(xTrain[0], xTrain[1]);
    axs.plot(xTest[0], xTest[1], linestyle='--');
    axs.axis('equal');
    plt.show();
