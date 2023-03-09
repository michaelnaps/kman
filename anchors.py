import numpy as np
import matplotlib.pyplot as plt

import Helpers.KoopmanFunctions as kman
from Helpers.KoopmanFunctions import vec
import Helpers.DataFunctions as data


# set global output setting
np.set_printoptions(precision=3, suppress=True);


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
        [0, 1]
    ] );
    B = np.array( [
        [dt, 0],
        [0, dt]
    ] );

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
    da = np.empty( (Na,Nx*Nx) );
    xa = np.empty( (Na,Nx*Nx) );
    ua = np.empty( (Na,Nu*Nx) );

    for i, a in enumerate(aList.T):
        a = a.reshape(Nx,1);
        da[i,:] = vec((x - a)@(x - a).T).reshape(1,Nx*Nx);
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
        meta['Nk'] = Nx+Nu+Nx*Nx+Nu*Nu+Nx*Nu;
        return meta;
    
    x = X[:Nx].reshape(Nx,1);
    u = X[Nx:].reshape(Nu,1);

    xx = vec(x@x.T);
    uu = vec(u@u.T);
    xu = vec(x@u.T);

    Psi = np.vstack( (x, u, xx, uu, xu) );

    return Psi;

def obsA(X=None):
    if X is None:
        meta = {
            'Nk': Nx+Nu+Nx*Nx+Nu*Nu+Nx*Nu + 1*Na*Nx*Nx+0*Na*Nu*Nx+1,
            'x':  [i for i in range(Nx)],
            'u':  [i for i in range(Nx, Nx+Nu)],
            'xx': [i for i in range(Nx+Nu, Nx+Nu+Nx*Nx)],
            'uu': [i for i in range(Nx+Nu+Nx*Nx, Nx+Nu+2*Nx*Nx)],
            'xu': [i for i in range(Nx+Nu+2*Nx*Nx, Nx+Nu+3*Nx*Nx)],
            # 'da': [i for i in range(Nx+Nu+3*Nx*Nx, Nx+Nu+3*Nx*Nx+1)],
            # 'xa': [i for i in range(Nx+Nu+3*Nx*Nx+1, Nx+Nu+3*Nx*Nx+1+Nx*Na)],
            # 'ua': [i for i in range(Nx+Nu+3*Nx*Nx+1+Nx*Na, Nx+Nu+3*Nx*Nx+1+Nx*Na+Nu*Na)]
        };
        return meta;

    x = X[:Nx].reshape(Nx,1);
    u = X[Nx:].reshape(Nu,1);
    a = aList;

    da, xa, ua = anchorExpand(x, u);   

    PsiX = obs(X);
    PsiA = np.vstack( (PsiX, da, [1]) );

    return PsiA;


# plot results
def plotcomp(xTest, PsiTest, save=0):
    # plot test results
    figRes, axsRes = plt.subplots();

    axsRes.plot(xTest[0], xTest[1], label='Model');
    axsRes.plot(PsiTest[0], PsiTest[1], linestyle='--', label='KCE');

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    axsRes.axis('equal');
    figRes.tight_layout();
    axsRes.legend();
    plt.grid();

    # show error
    Te = tList[0][-1];  Ne = round(Te/dt);
    figError, axsError = plt.subplots();

    axsError.plot([tList[0][0], tList[0][Ne]], [0,0], color='r', linestyle='--', label='ref');
    axsError.plot(tList[0][:Ne], PsiTest[0,:Ne]-xTest[0,:Ne], label='$x_1$');
    axsError.plot(tList[0][:Ne], PsiTest[1,:Ne]-xTest[1,:Ne], label='$x_2$');

    axsError.set_ylim( (-1,1) );
    axsError.grid();
    axsError.legend();

    # save results
    if save:
        figRes.savefig('/home/michaelnaps/prog/kman/.figures/uDonald.png', dpi=600);
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
    kvar = kman.KoopmanOperator(obsA, obs);
    K = kvar.edmd(X, Y, XU0);

    print('K:', K.shape, kvar.err)

    print(meta);
    print('Kx:\n', K[meta['x'],:].T);
    print('Ku:\n', K[meta['u'],:].T);
    print('Kxx:\n', K[meta['xx'],:].T);
    print('Kuu:\n', K[meta['uu'],:].T);
    print('Kxu:\n', K[meta['xu'],:].T);
    # print('Kda:\n', K[meta['da'],:].T);
    # print('Kxa:\n', K[meta['xa'],:].T);
    # print('Kua:\n', K[meta['ua'],:].T);


    # test results
    x0 = 5*np.random.rand(Nx,1) - 2.5;
    xu0 = np.vstack( (x0, [[0],[0]]) );
    Psi0 = obs(xu0);

    kModel = lambda Psi: K@rmes(Psi);
    def rmes(Psi):
        PsiA = obsA(Psi[:Nx+Nu].reshape(Nx+Nu,1));
        return PsiA;

    xTest, uTest = data.generate_data(tList, model, x0,
        control=control, Nu=Nu);
    PsiTest = data.generate_data(tList, kModel, Psi0)[0];


    # plot results
    plotcomp(xTest, PsiTest);


