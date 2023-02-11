import sys
sys.path.insert(0, '/home/michaelnaps/prog/ode');

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.path as path

import ode
import Helpers.KoopmanFunctions as kman


def modelFunc(x, _1, _2):
    return x - x**2;  # stable for x <= 1


def obsX(x=None):
    if x is None:
        meta = {'Nk':1};
        return meta;
    return x;

def obsH(x=None):
    if x is None:
        meta = {'Nk':3};
        return meta;
    return np.array( [[1], x, x**2] );

def obsU(x=None):
    if x is None:
        meta = {'Nk':1};
        return meta;
    return 1;


if __name__ == "__main__":
    # model parameters
    Nx = 1;
    x0 = np.array( [.90] );


    # model class variable
    model_type = 'discrete';
    mvar = ode.Model(modelFunc, model_type, x0=x0);


    # simulate model
    sim_time = 10;
    tTest, xTest = mvar.simulate(sim_time, x0)[:2];
    Nt = len(tTest);

    fig, axs = plt.subplots();
    axs.plot(tTest, xTest);


    # define observable function
    def obs(x=None):
        if x is None:
            meta = {'Nk': obsX()['Nk'] + obsU()['Nk']*obsH()['Nk']}
            return meta;

        PsiX = obsX(x);
        PsiU = obsU(x);
        PsiH = obsH(x);

        Psi = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );

        return Psi;


    # train cumulative Koopman operator

    print(xTest);
    X = xTest[:,:Nt-1];
    Y = xTest[:,1:Nt];

    kvar = kman.KoopmanOperator(obs);
    K = kvar.edmd(X, Y, x0);
