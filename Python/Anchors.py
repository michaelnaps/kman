import sys
sys.path.insert(0, '/home/michaelnaps/prog/ode');

import numpy as np
import matplotlib.pyplot as plt

import ode
import Helpers.KoopmanFunctions as kman


# create global "measurement" variables
Nu = 2;
Na = 4;
anchors = 20*np.random.rand(2,Na) - 10;


class Parameters:
    def __init__(self, x0,
                 fig=None, axs=None,
                 buffer_length=10, pause=1e-3,
                 color='k'):
        if axs is None and fig is None:
            self.fig, self.axs = plt.subplots();
        else:
            self.fig = fig;
            self.axs = axs;

        self.axs.set_xlim(-10.5, 10.5);
        self.axs.set_ylim(-10.5, 10.5);
        self.fig.tight_layout();

        # event variable
        self.x = x0;
        self.axs.plot(x0[0], x0[1], marker='.');

        # initialize buffer (trail)
        self.color = color;
        self.pause = pause;

    def update(self, t, x):
        self.axs.plot(x[0], x[1], marker='.');

        plt.show(block=0);
        plt.pause(self.pause);

        return self;

def modelFunc(x, u, params=None):
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
    return A@x + B@u;

def callbackFunc(T, x, u, mvar):
    return mvar.params.update(T, x);


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
        dist[i] = np.sqrt((anchor.T - xp).T @ (anchor.T - xp));

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


# set global output setting
np.set_printoptions(precision=3, suppress=True);


if __name__ == "__main__":
    # construct model
    Nx = 4;

    dt = 0.1;
    xg = np.zeros( (Nx,1) );

    C = [
        [10, 0, 2.5, 0],
        [0, 10, 0, 2.5]
    ];
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
        xTrain[:,i+1] = modelFunc(xTrain[:,i], uRand[:,i]);


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


    # create the remeasure function
    NkX = obsX()['Nk'];
    NkU = obsU()['Nk'];
    def rmes(Psi):
        x = Psi[:Nx];
        u = Psi[Nx:];

        PsiX = x;
        PsiU = 1;

        X = np.vstack( (x, np.zeros( (Nu,1) )) )
        PsiH = obsH(X);

        Psi = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );

        return Psi;


    # generate model variable from koopman function
    koopFunc = lambda Psi, _1, _2: K@rmes(Psi);

    Psi0 = obs(x0);
    params = Parameters(Psi0);

    kModel = ode.Model(koopFunc, 'discrete', callbackFunc, params, x0=Psi0, dt=dt);
    kModel.setMinTimeStep(0.1);

    sim_time = 10;
    kModel.simulate(sim_time, Psi0, callback=callbackFunc, output=1);
