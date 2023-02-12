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
    mu = 2;
    x = X[0];  dx = X[1];

    xn = X.reshape(Nx,1) + dt*np.array( [
        [dx],
        [mu*(1 - x**2)*dx - x]
    ] );

    return xn;

def obs(x=None):
    if x is None:
        meta = {'Nk':29};
        return meta;
    Psi = np.vstack( (
        np.exp(x), np.exp(x**2), np.exp(x[1]*x[0]**2),
        np.exp(x)*x, np.exp(x**2)*x, np.exp(x[1]*x[0]**2)*x,
        np.exp(x)*x*x, np.exp(x**2)*x*x, np.exp(x[1]*x[0]**2)*x*x,
        np.exp(x)*x*x*x, np.exp(x**2)*x*x*x, np.exp(x[1]*x[0]**2)*x*x*x,
        np.exp(x)*x*x*x*x, np.exp(x**2)*x*x*x*x, np.exp(x[1]*x[0]**2)*x*x*x*x
    ) );
    return Psi;


if __name__ == "__main__":
    # model parameters
    Nx = 2;
    N0 = 10;
    X0 = 2*np.random.rand(Nx,N0) - 1;

    # create model data
    T = 10;
    Nt = round(T/dt) + 1;

    tList = np.array( [[i*dt for i in range(Nt)]] );

    xTrain, _ = data.generate_data(tList, modelFunc, X0);

    # organize data
    xData = xTrain[:,:Nt-1];
    yData = xTrain[:,1:Nt];

    X = data.stack_data(xData, N0, Nx, Nt-1);
    Y = data.stack_data(yData, N0, Nx, Nt-1);

    kvar = kman.KoopmanOperator(obs);
    K = kvar.edmd(X, Y, X0);

    print('K:', kvar.err);
    print(K, '\n');

    # compare koopman to real model
    Nk = obs()['Nk'];
    Psi0 = obs(X0[:,0].reshape(Nx,1));
    xComp = xTrain[:Nx,:];

    koopFunc = lambda Psi: K@Psi;
    PsiTest, _ = data.generate_data(tList, koopFunc, Psi0);
    xTest = PsiTest[:Nx,:];

    # plot comparison
    fig, axs = plt.subplots();
    axs.plot(xTrain[0], xTrain[1]);
    axs.plot(xTest[0], xTest[1], linestyle='--');
    plt.show();
