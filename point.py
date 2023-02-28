import numpy as np
import matplotlib.pyplot as plt

import Helpers.KoopmanFunctions as kman
import Helpers.DataFunctions as data


# set global output setting
np.set_printoptions(precision=3, suppress=True);


# hyper paramter(s)
dt = 0.01;
Nx = 4;
Nu = 2;


# model is discrete (no ode needed)
def model(x, u):
    A = np.array( [
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ] );
    B = np.array( [
        [0, 0],
        [0, 0],
        [dt, 0],
        [0, dt]
    ] );

    xn = A@x.reshape(Nx,1) + B@u.reshape(Nu,1);

    return xn;

def control(x):
    C = np.array( [
        [10, 0, 2.5, 0],
        [0, 10, 0, 2.5]
    ] );
    xg = np.zeros( (Nx,1) );

    u = C@(xg - x.reshape(Nx,1));

    return u;


# observable functions PsiX, PsiU, PsiH
def obs(X=None):
    if X is None:
        meta = {'Nk':obsX()['Nk']+obsU()['Nk']*obsH()['Nk']};
        return meta;
    
    x = X[:Nx];
    u = X[Nx:];

    PsiX = obsX(x);
    PsiU = obsU(u);
    PsiH = obsH(X);
    PsiXU = np.vstack( (PsiX, np.kron(PsiU,PsiH)) );

    return PsiXU;

def obsX(x=None):
    if x is None:
        meta = {'Nk': Nx};
        return meta;
    PsiX = x;
    return PsiX;

def obsU(u=None):
    if u is None:
        meta = {'Nk': 1}
        return meta;
    PsiU = [1];
    return PsiU;

def obsH(X=None):
    if X is None:
        meta = {'Nk':Nx+Nu};
        return meta;
    PsiH = X;
    return PsiH;


if __name__ == "__main__":
    # simulation variables
    T = 10;  Nt = round(T/dt) + 1;
    tList = np.array( [ [i*dt for i in range(Nt)] ] );


    # generate the randomized control policy
    randControl = lambda x: np.random.rand(Nu,1);


    # generate training data for Kx
    N0 = 10;
    X0 = 20*np.random.rand(Nx,N0) - 10;

    xData, uData = data.generate_data(tList, model, X0,
        control=control, Nu=Nu);


    # construct training data from xData and uData
    uStack = data.stack_data(uData, N0, Nu, Nt-1);
    xStack = data.stack_data(xData[:,:-1], N0, Nx, Nt-1);
    yStack = data.stack_data(xData[:,1:], N0, Nx, Nt-1);

    XU0 = np.vstack( (X0, np.zeros( (Nu,N0) )) );
    X = np.vstack( (xStack, uStack) );
    Y = np.vstack( (yStack, uStack) );


    # solve for Kx from data
    kxvar = kman.KoopmanOperator(obs);
    Kx = kxvar.edmd(X, Y, XU0);

    print('Kx:', Kx.shape, kxvar.err)
    print(Kx);
    print('\n');


    # construct data for Ku
    randModel = lambda x, u: np.random.rand(Nx,1);
    xRand, uTrain = data.generate_data(tList, randModel, X0,
        control=control, Nu=Nu);

    uStack = data.stack_data(uTrain, N0, Nu, Nt-1);
    xStack = data.stack_data(xRand[:,:-1], N0, Nx, Nt-1);


    # solve for Ku
    Xu = np.vstack( (xStack, np.zeros( (Nu,N0*(Nt-1)) )) );
    Yu = np.vstack( (xStack, uStack) );

    kuvar = kman.KoopmanOperator(obsH);
    Ku = kuvar.edmd(Xu, Yu, XU0)

    print('Ku:', Ku.shape, kuvar.err)
    print(Ku);
    print('\n');


    # generate cumulate operator
    m = Nu;
    p = obsX()['Nk'];
    q = obsU()['Nk'];
    b = obsH()['Nk'];

    Ktemp = np.vstack( (
        np.hstack( (np.eye(p), np.zeros( (p,q*b) )) ),
        np.hstack( (np.zeros( (b*q, p) ), np.kron(np.eye(q), Ku)) )
    ) );

    K = Kx @ Ktemp;

    print('K\n', K, '\n')


    # test the cumulative operator
    kModel = lambda Psi: K@rmes(Psi);
    def rmes(Psi):
        Nkx = obsX()['Nk'];
        Nku = obsU()['Nk'];
        Nkh = obsH()['Nk'];

        PsiX = Psi[:Nkx].reshape(Nkx,1);
        PsiU = [1];
        PsiH = obsH(Psi[Nkx:].reshape(Nku*Nkh,1));

        Psin = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );

        return Psin;

    x0 = np.array( [[3],[-1.4],[3],[-10],[0],[0]] );
    Psi0 = obs(x0);

    xTest = data.generate_data(tList, model, x0[:Nx].reshape(Nx,1), control=control, Nu=Nu)[0];
    PsiTest = data.generate_data(tList, kModel, Psi0)[0];

    
    # plot test results
    fig, axs = plt.subplots();
    axs.plot(xTest[0], xTest[1], label='Model');
    axs.plot(PsiTest[0], PsiTest[1], linestyle='--', label='KCE');

    plt.title('$x_0 = [3, -1.4, 3, -10]^\intercal$');
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    axs.axis('equal');
    plt.grid();
    
    fig.tight_layout();
    plt.legend();
    plt.show();
    