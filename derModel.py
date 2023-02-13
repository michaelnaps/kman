import sys
sys.path.insert(0, '/home/michaelnaps/prog/ode');

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.path as path

import ode
import Helpers.KoopmanFunctions as kman
import Helpers.DataFunctions as data


def modelFunc(x):
    return x - x**2;  # stable for x <= 1


# define observable function FOR TRAINING
def obs(x=None):
    if x is None:
        meta = {'Nk': obsX()['Nk'] + obsU()['Nk']*obsH()['Nk']}
        return meta;

    PsiX = obsX(x);
    PsiU = obsU(x);
    PsiH = obsH(x);

    Psi = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );

    return Psi;

def obsX(x=None):
    if x is None:
        meta = {'Nk':6};
        return meta;
    Psi = np.vstack( (
        np.exp(x), np.exp(x**2),
        np.exp(x)*x, np.exp(x**2)*(x**2),
        np.exp(x)*x*x, np.exp(x**2)*(x**2)*(x**2)
    ) );
    return Psi;

def obsH(x=None):
    if x is None:
        meta = {'Nk':1};
        return meta;
    return np.array( [1] );

def obsU(x=None):
    if x is None:
        meta = {'Nk':1};
        return meta;
    return np.array( [1] );


# set global output setting
np.set_printoptions(precision=3, suppress=True);


if __name__ == "__main__":
    # model parameters
    dt = 0.01;
    Nx = 1;
    x0 = np.random.rand(Nx,1);


    # simulate model
    T = 10;  Nt = round(T/dt) + 1;
    tList = np.array( [[i*dt for i in range(Nt)]] );

    xTrain, _ = data.generate_data(tList, modelFunc, x0);

    fig, axs = plt.subplots();
    axs.plot(tList[0], xTrain[0]);


    # train cumulative Koopman operator
    X = xTrain[:,:Nt-1];
    Y = xTrain[:,1:Nt];

    kvar = kman.KoopmanOperator(obs);
    Nk = kvar.meta['Nk'];
    K = kvar.edmd(X, Y, x0);

    print('K\n\n', K);

    Kup = K[:obsX()['Nk'],:];
    print('Kup\n', Kup);


    # model the koopman with updating observables
    def koopModel(Psi):
        Nk = obsX()['Nk'] + 1;
        Psi_n = K@Psi.reshape(Nk,1);
        return Psi_n.reshape(Nk,);

    Psi0 = obs(x0);
    PsiTest, _ = data.generate_data(tList, koopModel, Psi0);

    xTest = np.log(PsiTest[0]);

    axs.plot(tList[0], xTest, linestyle='--');
    plt.show();
