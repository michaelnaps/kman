import numpy as np
import matplotlib.pyplot as plt

from Helpers.KoopmanFunctions import *
import Helpers.DataFunctions as data


# set global output setting
np.set_printoptions(precision=5, suppress=True);


# hyper paramter(s)
dt = 0.01;
Nx = 2;
Nu = 2;
Na = 3;
aList = np.array( [[10, 10, -10],[10, -10, -10]] );


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

def anchorExpand(x, u):
    da = np.empty( (Na,1) );
    xa = np.empty( (Na,Nx*Nx) );
    ua = np.empty( (Na,Nu*Nx) );

    for i, a in enumerate(aList.T):
        a = a.reshape(Nx,1);
        da[i,:] = vec((x - a).T@(x - a));
        xa[i,:] = vec(x@a.T).reshape(1,Nx*Nx);
        ua[i,:] = vec(u@a.T).reshape(1,Nu*Nx);

    da = vec(da);
    xa = vec(xa);
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
        meta = {'Nk':Nu+Na};
        return meta;

    x = X[:Nx].reshape(Nx,1);
    u = X[Nx:].reshape(Nu,1);

    da = anchorExpand(x, u)[0];
    PsiH = np.vstack( (u, da) );

    return PsiH;

def obsXUH(X=None):
    if X is None:
        meta = {'Nk':obsX()['Nk']+obsU()['Nk']*obsX()['Nk']}
        return meta;

    PsiX = obsX(X);
    PsiU = [1];
    PsiH = obsH(X);

    Psi = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );

    return Psi;


def ploterr(X, Y, X0, save=0):
    # show error
    Te = tList[0][-1];  Ne = round(Te/dt);
    figError, axsError = plt.subplots();

    axsError.plot([tList[0][0], tList[0][Ne]], [0,0], color='r', linestyle='--', label='ref');
    axsError.plot(tList[0][:Ne], Y[0,:Ne]-X[0,:Ne], label='$x_1$');
    axsError.plot(tList[0][:Ne], Y[1,:Ne]-X[1,:Ne], label='$x_2$');

    axsError.set_ylim( (-1,1) );
    axsError.grid();
    axsError.legend();

    # save results
    if save:
        figError.savefig('/home/michaelnaps/prog/kman/.figures/uDonaldError.png', dpi=600);
    else:
        plt.show();


# main executable section
if __name__ == "__main__":
    # simulation variables
    T = 10;  Nt = round(T/dt) + 1;
    tList = np.array( [ [i*dt for i in range(Nt)] ] );


    # generate training data for Kx
    N0 = 1;
    X0 = 10*np.random.rand(Nx,N0) - 5;

    randControl = lambda x: 2*np.random.rand(Nu,1)-1;
    xData, uRand = data.generate_data(tList, model, X0,
        control=randControl, Nu=Nu);

    # formatting training data from xData and uData
    uStack = data.stack_data(uRand, N0, Nu, Nt-1);
    xStack = data.stack_data(xData[:,:-1], N0, Nx, Nt-1);
    yStack = data.stack_data(xData[:,1:], N0, Nx, Nt-1);

    XU0 = np.vstack( (X0, np.zeros( (Nu,N0) )) );
    X = np.vstack( (xStack, uStack) );
    Y = np.vstack( (yStack, uStack) );

    # train Kx
    metaX = obsXU();
    kxvar = KoopmanOperator(obsXU);
    Kx = kxvar.edmd(X, Y, XU0);

    print('Kx:', Kx.shape, kxvar.err)
    print(Kx);


    # generate data for Ku
    randModel = lambda x,u: 10*np.random.rand(Nx,1)-5;
    xRand, uData = data.generate_data(tList, randModel, X0,
        control=control, Nu=Nu);

    uStack = data.stack_data(uData, N0, Nu, Nt-1);
    xStack = data.stack_data(xRand[:,:-1], N0, Nx, Nt-1);

    Xu = np.vstack( (xStack, np.zeros( (Nu,N0*(Nt-1)) )) );
    Yu = np.vstack( (xStack, uStack) );

    # train Ku
    metaH = obsH();
    kuvar = KoopmanOperator(obsH, obsU);
    print(kuvar);
    Ku = kuvar.edmd(Xu, Yu, XU0);

    print('Ku:', Ku.shape, kuvar.err);
    print(Ku);

    # generate cumulative operator
    m = Nu;
    p = obsX()['Nk'];
    q = obsU()['Nk'];
    b = obsH()['Nk'];

    Ktemp = np.vstack( (
        np.hstack( (np.eye(p), np.zeros( (p,b) )) ),
        np.hstack( (np.zeros( (m,p) ), Ku) ) ) );

    K = Kx@Ktemp;

    print('K:');
    print(K);


    # test comparison results
    N0n = 10;
    NkXU = obsXU()['Nk'];
    X0n = 20*np.random.rand(Nx,N0n) - 10;
    XU0n = np.vstack( (X0n, np.zeros( (Nu,N0n) )) );
    
    Psi0 = np.empty( (NkXU,N0n) );
    for i, xu in enumerate(XU0n.T):
        Psi0[:,i] = obsXU( xu.reshape(Nx+Nu,1) ).reshape(NkXU,);

    # new operator model equation
    kModel = lambda Psi: K@rmes(Psi);
    def rmes(Psi):
        x = Psi[:Nx].reshape(Nx,1);
        u = np.zeros( (Nu,1) );

        PsiX = Psi[:p].reshape(p,1);
        PsiU = [1];
        PsiH = obsH(Psi.reshape(p+q,1));

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
    plt.show();

