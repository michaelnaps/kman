import numpy as np
import matplotlib.pyplot as plt

from Helpers.KoopmanFunctions import *
import Helpers.DataFunctions as data


# set global output setting
np.set_printoptions(precision=5, suppress=True);


# hyper paramter(s)
eps = 0.1;
dt = 0.01;
Nx = 2;
Nu = 2;
Na = 3;
aList = np.array( [[10, 10, -10],[10, -10, -10]] );


# plot functions
def plotAnchors(fig, axs):
    for a in aList.T:
        axs.plot(a[0], a[1]);  # to shape axis around anchors
        circEntity = plt.Circle(a, 0.5, facecolor='indianred', edgecolor='k');
        axs.add_artist( circEntity );
    # axs.axis('equal');
    return;


# model and control functions
def model(x, u):
    A = np.array( [
        [1, 0],
        [0, 1] ] );
    B = np.array( [
        [dt, 0],
        [0, dt] ] );

    xn = A@x.reshape(Nx,1) + B@u.reshape(Nu,1);

    return xn;

def control(x):
    C = np.array( [
        [1, 0],
        [0, 1]
    ] );
    xg = np.zeros( (Nx,1) );

    u = C@(xg - x.reshape(Nx,1));

    return u;

def noise(eps, shape):
    return eps*np.random.rand(shape[0], shape[1]) - eps/2;

def measure(x):
    d = anchorExpand(x);
    d += noise(eps, d.shape);
    return d;

def anchorExpand(x, u=None):
    da = np.empty( (Na,1) );
    xa = np.empty( (Na,Nx*Nx) );
    if u is not None:
        ua = np.empty( (Na,Nu*Nx) );
    else:
        ua = None;

    for i, a in enumerate(aList.T):
        a = a.reshape(Nx,1);
        da[i,:] = vec((x - a).T@(x - a));
        xa[i,:] = vec(x@a.T).reshape(1,Nx*Nx);
        if u is not None:
            ua[i,:] = vec(u@a.T).reshape(1,Nu*Nx);

    da = vec(da);
    xa = vec(xa);
    if u is not None:
        ua = vec(ua);

    return da, xa, ua;


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

def obsXU(X=None):
    if X is None:
        meta = {'Nk':obsX()['Nk']+obsU()['Nk']};
        return meta;
    PsiX = obsX(X);
    PsiU = obsU(X);
    PsiXU = np.vstack( (PsiX, PsiU) );
    return PsiXU;

def obsH(X=None):
    if X is None:
        meta = {'Nk':Na};
        return meta;

    x = X[:Nx].reshape(Nx,1);
    u = X[Nx:].reshape(Nu,1);

    da = anchorExpand(x, u)[0];
    PsiH = da;

    return PsiH;

def obsXUH(X=None):
    if X is None:
        meta = {'Nk':obsX()['Nk']+1*obsH()['Nk']}
        return meta;

    PsiX = obsX(X);
    PsiU = [1];
    PsiH = obsH(X);

    Psi = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );

    return Psi;


# main executable section
if __name__ == "__main__":
    # simulation variables
    T = 1;  Nt = round(T/dt) + 1;
    tList = np.array( [ [i*dt for i in range(Nt)] ] );

    # dimensiones reference variables
    m = Nu;
    p = obsX()['Nk'];
    q = 1;
    b = obsH()['Nk'];

    # generate training data for Kx
    N0 = 2;
    X0 = 10*np.random.rand(Nx,N0) - 5;
    xData, uRand = data.generate_data(tList, model, X0,
        control=control, Nu=Nu);

    # formatting training data from xData and uData
    uStack = data.stack_data(uRand, N0, Nu, Nt-1);
    xStack = data.stack_data(xData[:,:-1], N0, Nx, Nt-1);
    yStack = data.stack_data(xData[:,1:], N0, Nx, Nt-1);

    XU0 = np.vstack( (X0, np.zeros( (Nu,N0) )) );
    X = np.vstack( (xStack, uStack) );
    Y = np.vstack( (yStack, uStack) );

    # Ku block diagonal matrix function
    def Mu(kvar):
        Kblock = np.vstack( (
            np.hstack( (np.eye(p), np.zeros( (p,b*q) )) ),
            np.hstack( (np.zeros( (m,p) ), np.kron( np.eye(q), kvar.K)) )
        ) );
        return Kblock;

    # initialize operator variables and solve
    kuvar = KoopmanOperator(obsH, obsU);
    kxvar = KoopmanOperator(obsXUH, obsXU, M=Mu(kuvar));

    klist = (kxvar, kuvar);
    mlist = (Mu, );
    klist = cascade_edmd(klist, mlist, X, Y, XU0);

    # form the cumulative operator
    Kfinal = klist[0].K @ Mu( klist[1] );
    kvar = KoopmanOperator(obsXUH, obsXU, K=Kfinal);

    kvar.resError(X, Y, XU0);
    for k in klist:
        print(k);
    print(kvar);

    # test comparison results
    N0n = 25;
    NkXU = obsXU()['Nk'];
    X0n = 20*np.random.rand(Nx,N0n) - 10;
    XU0n = np.vstack( (X0n, np.zeros( (Nu,N0n) )) );

    Psi0 = np.empty( (NkXU,N0n) );
    for i, xu in enumerate(XU0n.T):
        Psi0[:,i] = obsXU( xu.reshape(Nx+Nu,1) ).reshape(NkXU,);

    # new operator model equation
    kModel = lambda Psi: kvar.K@rmes(Psi);
    def rmes(Psi):
        x = Psi[:Nx].reshape(Nx,1);
        u = np.zeros( (Nu,1) );

        PsiX = Psi[:p].reshape(p,1);
        PsiU = [1];
        PsiH = obsH(Psi);

        Psin = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );

        return Psin;

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
    plotAnchors(figComp, axsComp);
    plt.show();
