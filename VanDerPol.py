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
    mu = 0;
    x = X[0];  dx = X[1];

    xn = X.reshape(Nx,1) + dt*np.array( [
        [dx],
        [mu*(dx - dx*x**2) - x]
    ] );

    return xn;

def obs(x=None):
    if x is None:
        meta = {'Nk':25};
        return meta;

    z1 = x;
    z2 = x**2;
    z3 = x[1]*x[0]**2;

    Psi = np.vstack( (
        np.exp(z1), np.exp(z2), np.exp(z3),
        np.exp(z1)*z1, np.exp(z2)*z2, np.exp(z3)*z3,
        np.exp(z1)*z1*z1, np.exp(z2)*z2*z2, np.exp(z3)*z3*z3,
        np.exp(z1)*z1*z1*z1, np.exp(z2)*z2*z2*z2, np.exp(z3)*z3*z3*z3,
        np.exp(z1)*z1*z1*z1*z1, np.exp(z2)*z2*z2*z2*z2, np.exp(z3)*z3*z3*z3*z3
    ) );

    return Psi;


if __name__ == "__main__":
    # model parameters
    Nx = 2;
    N0 = 30;
    X0 = 2*np.random.rand(Nx,N0) - 1;

    # create model data
    T = 50;
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
    axs.axis('equal');
    plt.show();
