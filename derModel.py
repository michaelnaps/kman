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
    x0 = np.array( [0.90] );


    # model class variable
    model_type = 'discrete';
    mvar = ode.Model(modelFunc, model_type, x0=x0, dt=dt);
    mvar.setMinTimeStep(0.01);


    # simulate model
    sim_time = 10;
    tList, xTrain = mvar.simulate(sim_time, x0)[:2];
    Nt = tList.shape[1];

    fig, axs = plt.subplots();
    axs.plot(tList[0], xTrain[0]);


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
    X = xTrain[:,:Nt-1];
    Y = xTrain[:,1:Nt];

    kvar = kman.KoopmanOperator(obsTrain);
    Nk = kvar.meta['Nk'];
    K = kvar.edmd(X, Y, x0);

    print('K\n\n', K);

    Kup = K[:obsX()['Nk'],:];
    print('Kup\n', Kup);


    # model the koopman with updating observables
    def koopModel(Psi, _1, _2):
        Nk = obsX()['Nk'] + 1;
        Psi_n = K@Psi.reshape(Nk,1);
        return Psi_n.reshape(Nk,);

    Psi0 = obsTrain(x0);
    koopModelVar = ode.Model(koopModel, model_type, x0=Psi0, dt=dt);
    PsiTest = koopModelVar.simulate(sim_time, Psi0)[1];

    xTest = np.log(PsiTest[0]);

    axs.plot(tList[0], xTest, linestyle='--');
    plt.show();
