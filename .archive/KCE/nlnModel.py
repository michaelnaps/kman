import sys
sys.path.insert(0, '/home/michaelnaps/prog/ode');

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.path as path

import ode
import KMAN.KoopmanFunctions as kman
import KMAN.DataFunctions as data


# hyper parameter(s)
dt = 0.01;


def modelFunc(x):
    return x - x**2;  # stable for x <= 1


def obsX(x=None):
    if x is None:
        meta = {'Nk':2};
        return meta;
    return np.vstack( (x, x**2) );

def obsH(x=None):
    if x is None:
        meta = {'Nk':1};
        return meta;
    return x**2;

def obsU(x=None):
    if x is None:
        meta = {'Nk':obsX()['Nk']};
        return meta;
    return obsX(x);


# set global output setting
np.set_printoptions(precision=3, suppress=True);


if __name__ == "__main__":
    # model parameters
    Nx = 1;
    x0 = np.random.rand(Nx,1);


    # simulate model
    T = 10;  Nt = round(T/dt) + 1;
    tList = np.array( [[i*dt for i in range(Nt)]] )

    xTest, _ = data.generate_data(tList, modelFunc, x0);


    # define observable function FOR TRAINING
    def obsTrain(x=None):
        if x is None:
            meta = {'Nk': obsX()['Nk'] + obsU()['Nk']*obsH()['Nk']}
            return meta;

        PsiX = obsX(x);
        PsiU = obsU(x);
        PsiH = obsH(x);

        Psi = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );

        return Psi;


    # train cumulative Koopman operator
    X = xTest[:,:Nt-1];
    Y = xTest[:,1:Nt];

    kvar = kman.KoopmanOperator(obsTrain);
    Nk = kvar.meta['Nk'];
    K = kvar.edmd(X, Y, x0);

    print('K\n\n', K);

    Kup = K[:obsX()['Nk'],:];
    print('Kup\n', Kup);


    # define observable function FOR IMPLEMENTATION
    def obsImplm(X=None):
        if X is None:
            meta = {'Nk': 4};
            return meta;
        PsiH = X[1];
        Psi = np.vstack( (X, np.kron(X, PsiH)) );
        return Psi.reshape(4,);


    # model the koopman with updating observables
    koopModel = lambda x: Kup@obsImplm(x);

    Psi0 = obsX(x0);
    PsiTest, _ = data.generate_data(tList, koopModel, Psi0);


    # plot result comparisons
    fig, axs = plt.subplots();
    axs.plot(tList[0], xTest[0]);
    axs.plot(tList[0], PsiTest[0], linestyle='--');
    plt.show();
