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
def obs(X=None):
    if X is None:
        meta = obsA();
        meta['Nk'] = Nx+Nu+0*Nx*Nx+0*Nu*Nu+0*Nx*Nu;
        return meta;
    
    x = X[:Nx].reshape(Nx,1);
    u = X[Nx:].reshape(Nu,1);

    # xx = vec(x@x.T);
    # uu = vec(u@u.T);
    # xu = vec(x@u.T);

    Psi = np.vstack( (x, u) );

    return Psi;

def obsA(X=None):
    if X is None:
        meta = {
            'Nk': Nx+Nu+0*Nx*Nx+0*Nu*Nu+0*Nx*Nu + 1*Na+0*Na*Nu*Nx+1,
            'x':  [i for i in range(Nx)],
            'u':  [i for i in range(Nx, Nx+Nu)],
            # 'xx': [i for i in range(Nx+Nu, Nx+Nu+Nx*Nx)],
            # 'uu': [i for i in range(Nx+Nu+Nx*Nx, Nx+Nu+2*Nx*Nx)],
            # 'xu': [i for i in range(Nx+Nu+2*Nx*Nx, Nx+Nu+3*Nx*Nx)],
            # 'da': [i for i in range(Nx+Nu+3*Nx*Nx, Nx+Nu+3*Nx*Nx+1)],
            # 'xa': [i for i in range(Nx+Nu+3*Nx*Nx+1, Nx+Nu+3*Nx*Nx+1+Nx*Na)],
            # 'ua': [i for i in range(Nx+Nu+3*Nx*Nx+1+Nx*Na, Nx+Nu+3*Nx*Nx+1+Nx*Na+Nu*Na)]
        };
        return meta;

    x = X[:Nx].reshape(Nx,1);
    u = X[Nx:].reshape(Nu,1);

    da, xa, ua = anchorExpand(x, u);   

    PsiX = obs(X);
    PsiA = np.vstack( (PsiX, da, [1]) );

    return PsiA;


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
    N0 = 5;
    X0 = 10*np.random.rand(Nx,N0) - 5;

    print(X0);

    xData, uData = data.generate_data(tList, model, X0,
        control=control, Nu=Nu);


    # formatting training data from xData and uData
    uStack = data.stack_data(uData, N0, Nu, Nt-1);
    xStack = data.stack_data(xData[:,:-1], N0, Nx, Nt-1);
    yStack = data.stack_data(xData[:,1:], N0, Nx, Nt-1);

    XU0 = np.vstack( (X0, np.zeros( (Nu,N0) )) );
    X = np.vstack( (xStack, np.zeros( (Nu,N0*(Nt-1)) )) );
    Y = np.vstack( (yStack, uStack) );


    # train for operator
    meta = obsA();
    kvar = KoopmanOperator(obsA, obs);
    K = kvar.edmd(X, Y, XU0);

    print('K:', K.shape, kvar.err)
    print(K);

    print(meta);
    print('Kx:\n', K[meta['x'],:].T);
    print('Ku:\n', K[meta['u'],:].T);


    # for reference
    Nk  = obs()['Nk'];
    NkA = obsA()['Nk'];


    # test comparison results
    N0n = 10;
    X0n = 20*np.random.rand(Nx,N0n) - 10;
    XU0n = np.vstack( (X0n, np.zeros( (Nu,N0n) )) );
    
    Psi0 = np.empty( (Nk,N0n) );
    for i, xu in enumerate(XU0n.T):
        Psi0[:,i] = obs( xu.reshape(Nx+Nu,1) ).reshape(Nk,);

    # new operator model equation
    kModel = lambda Psi: K@rmes(Psi);
    def rmes(Psi):
        # should be dependent on "real" measurement - not Psi
        PsiA = obsA(Psi[:Nx+Nu].reshape(Nx+Nu,1));
        return PsiA;

    xTest, uTest = data.generate_data(tList, model, X0n,
        control=control, Nu=Nu);
    PsiTest, _ = data.generate_data(tList, kModel, Psi0);

    # plot results
    xPsi = np.empty( (N0n*Nx, Nt) );
    i = 0;  j = 0;
    for k in range(N0n):
        xPsi[i:i+Nx,:] = PsiTest[j:j+Nx,:];
        i += Nx;
        j += Nk;
    figComp, axsComp = data.compare_data(xTest, xPsi, X0n);
    plt.show();

