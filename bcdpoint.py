import numpy as np
import matplotlib.pyplot as plt

from Helpers.KoopmanFunctions import *
import Helpers.DataFunctions as data


# set global output setting
np.set_printoptions(precision=3, suppress=True);


# hyper paramter(s)
dt = 0.01;
Nx = 4;
Nu = 2;


# model and control functions
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
        [10, 0, 5, 0],
        [0, 10, 0, 5]
    ] );
    xg = np.zeros( (Nx,1) );

    u = C@(xg - x.reshape(Nx,1));

    return u;


# observable functions PsiX, PsiU, PsiH
def obsX(X=None):
    if X is None:
        meta = {'Nk':Nx};
        return meta;
    PsiX = X[:Nx].reshape(Nx,1);
    return PsiX;

def obsU(X=None):
    if X is None:
        meta = {'Nk':Nu};
        return meta;
    PsiU = X[Nx:].reshape(Nu,1);
    return PsiU;

def obsXU(X=None):  # proabably don't need
    if X is None:
        meta = {'Nk':obsX()['Nk']+obsH()['Nk']};
        return meta;

    PsiX = obsX(X);
    PsiU = [1];
    Psi = np.vstack( (PsiX, PsiU) );

    return Psi;

def obsH(X=None):
    if X is None:
        meta = {'Nk':Nx+Nu};
        return meta;
    PsiH = X;
    return PsiH;

def obsXUH(X=None):
    if X is None:
        meta = {'Nk':obsX()['Nk']+1*obsH()['Nk']};
        return meta;
    
    PsiX = obsX(X);
    PsiU = [1];
    PsiH = obsH(X);
    
    Psi = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );
    return Psi;


# main executable section
if __name__ == "__main__":
    # simulation variables
    T = 10;  Nt = round(T/dt) + 1;
    tList = np.array( [ [i*dt for i in range(Nt)] ] );

    # generate the randomized control policy
    randControl = lambda x: np.random.rand(Nu,1);

    # generate training data for Kx
    N0 = 2;
    X0 = 20*np.random.rand(Nx,N0) - 10;

    # construct training data from xData and uData
    xData, uData = data.generate_data(tList, model, X0,
        control=control, Nu=Nu);

    uStack = data.stack_data(uData, N0, Nu, Nt-1);
    xStack = data.stack_data(xData[:,:-1], N0, Nx, Nt-1);
    yStack = data.stack_data(xData[:,1:], N0, Nx, Nt-1);

    XU0 = np.vstack( (X0, np.zeros( (Nu,N0) )) );
    X = np.vstack( (xStack, np.zeros( (Nu, N0*(Nt-1)) )) );
    Y = np.vstack( (yStack, uStack) );

    # matrices dimensions
    m = Nu;
    p = obsX()['Nk'];
    q = 1; # obsU()['Nk'];
    b = obsH()['Nk'];

    # construct matrices functions
    def Kblock(K):
        Kb = np.vstack( (
            np.hstack( (np.eye(p), np.zeros( (p, b*q) )) ),
            np.hstack( (np.zeros( (b*q, p) ), np.kron(np.eye(q), K)) )
        ) );
        return Kb;
    def Mx(Klist, G):
        # M = np.kron( G.T@Klist[1].K.T, np.eye(p+b*q) );
        M = np.kron( G.T@Kblock(Klist[1].K).T, np.eye(p+b*q) );
        return M;
    def Mu(Klist, G):
        M = np.kron( G.T, np.eye(b) );
        return M;

    # initialize operator class (K0 is identity)
    kuvar = KoopmanOperator(obsH);
    kxvar = KoopmanOperator(obsXUH);
    kxvar, kuvar = bcd( (kxvar,kuvar), (Mx,Mu), X, Y, X0 );

    K = kxvar.K@Kblock(kuvar.K);

    print(kxvar, '\n');
    print(kuvar, '\n');
    print(K);


    # # new operator model equation
    # kModel = lambda Psi: K@Psi;

    # # data for testing results
    # N0n = 25;
    # X0n = 20*np.random.rand(Nx,N0n) - 10;
    # XU0n = np.vstack( (X0n, np.zeros( (Nu,N0n) )) );
    
    # Psi0 = np.empty( (Nk,N0n) );
    # for i, xu in enumerate(XU0n.T):
    #     Psi0[:,i] = obs( xu.reshape(Nx+Nu,1) ).reshape(Nk,);

    # xTest, uTest = data.generate_data(tList, model, X0n,
    #     control=control, Nu=Nu);
    # PsiTest, _ = data.generate_data(tList, kModel, Psi0);

    # # plot results
    # xPsi = np.empty( (N0n*Nx, Nt) );
    # i = 0;  j = 0;
    # for k in range(N0n):
    #     xPsi[i:i+Nx,:] = PsiTest[j:j+Nx,:];
    #     i += Nx;
    #     j += Nk;
    # figComp, axsComp = data.compare_data(xTest, xPsi, X0n);
    # plt.show();
    