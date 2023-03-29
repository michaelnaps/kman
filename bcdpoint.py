import numpy as np
import matplotlib.pyplot as plt

from Helpers.KoopmanFunctions import *
import Helpers.DataFunctions as data


# set global output setting
np.set_printoptions(precision=2, suppress=True);


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
        meta = {'Nk':Nx+Nu};
        return meta;
    PsiXU = X;
    return PsiXU;

def obsH(X=None):
    if X is None:
        meta = {'Nk':Nx+Nu};
        return meta;
    PsiH = X;
    return PsiH;

def obsXUH(X=None):
    NkX = obsX()['Nk'];
    NkU = 1;
    NkH = obsH()['Nk'];
    if X is None:
        meta = {'Nk':NkX+NkU*NkH};
        return meta;
    
    PsiX = obsX(X).reshape(NkX,1);
    PsiU = [1];
    PsiH = obsH(X).reshape(NkH,1);
    
    Psi = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );
    return Psi;


# main executable section
if __name__ == "__main__":
    # simulation variables
    T = 10;  Nt = round(T/dt) + 1;
    tList = np.array( [ [i*dt for i in range(Nt)] ] );
    # print(tList);

    # generate the randomized control policy
    randControl = lambda x: np.random.rand(Nu,1);

    # generate training data for Kx
    N0 = 2;
    X0 = 2*np.random.rand(Nx,N0) - 1;

    # construct training data from xData and uData
    xData, uData = data.generate_data(tList, model, X0,
        control=control, Nu=Nu);

    XU0 = np.vstack( (X0, np.zeros( (Nu,N0) )) );
    xStack = data.stack_data(xData[:,:-1], N0, Nx, Nt-1);
    yStack = data.stack_data(xData[:,1:], N0, Nx, Nt-1);
    uStack = data.stack_data(uData, N0, Nu, Nt-1);

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
            np.hstack( (np.zeros( (m, p) ), np.kron(np.eye(q), K[-m:,:])) )
        ) );
        return Kb;

    def Mx(Klist):
        M = Kblock( Klist[0].K );
        return M;

    # initialize operator class (K0 is identity)
    kuvar = KoopmanOperator(obsH);
    kxvar = KoopmanOperator(obsXUH, obsXU, M=Mx( (kuvar, None) ));
    
    klist = (kuvar, kxvar);
    mlist = (None, Mx);
    klist = bcd( klist, mlist, X, Y, XU0 );

    # initialize and check cumulative Koopman operator
    Kvar = KoopmanOperator(obsXUH, obsXU, K=kxvar.K@Kblock(kuvar.K));
    Kvar.resError(X, Y, XU0);

    for kvar in klist:
        print(kvar, '\n');
    print(Kvar);


    # new operator model equation
    NkXU = obsXU()['Nk'];
    NkXUH = obsXUH()['Nk'];
    kModel = lambda Psi: Kvar.K@obsXUH(Psi);

    # data for testing results
    N0n = 10;
    X0n = 20*np.random.rand(Nx,N0n) - 10;
    XU0n = np.vstack( (X0n, np.zeros( (Nu,N0n) )) );
    
    Psi0 = np.empty( (NkXU,N0n) );
    for i, xu in enumerate(XU0n.T):
        Psi0[:,i] = obsXU( xu.reshape(Nx+Nu,1) ).reshape(NkXU,);

    xTest, uTest = data.generate_data(tList, model, X0n,
        control=control, Nu=Nu);
    PsiTest, _ = data.generate_data(tList, kModel, Psi0);

    # plot results
    xPsi = np.empty( (N0n*Nx, Nt) );
    i = 0;  j = 0;
    for k in range(N0n):
        xPsi[i:i+Nx,:] = PsiTest[j:j+Nx,:];
        i += Nx;
        j += NkXU;
    figComp, axsComp = data.compare_data(xTest, xPsi, X0n);

    save = 0;
    if save:
        plt.savefig('.figures/point.png', dpi=600);
    else:
        plt.show();

    