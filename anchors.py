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
Na = 2;
aList = np.array( [[1],[1]] );


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
        [10, 0],
        [0, 10]
    ] );
    xg = np.zeros( (Nx,1) );

    u = C@(xg - x.reshape(Nx,1));

    return u;

def anchorDist(x):
    d = np.empty( (Na,1) );

    for i, a in enumerate(aList.T):
        a = a.reshape(Nx,1);
        d[i] = (x - a).T@(x - a);

    return d;


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
            'Nk': Nx+Nu+Nx*Nx+Nu*Nu+Nx*Nu + 2*Nx*Na+Nu*Na,
            'x':  [i for i in range(Nx)],
            'u':  [i for i in range(Nx, Nx+Nu)],
            'xx': [i for i in range(Nx+Nu, Nx+Nu+Nx*Nx)],
            'uu': [i for i in range(Nx+Nu+Nx*Nx, Nx+Nu+2*Nx*Nx)],
            'xu': [i for i in range(Nx+Nu+2*Nx*Nx, Nx+Nu+3*Nx*Nx)],
            'da': [i for i in range(Nx+Nu+3*Nx*Nx, Nx+Nu+3*Nx*Nx+Nx*Na)],
            'xa': [i for i in range(Nx+Nu+3*Nx*Nx+Nx*Na, Nx+Nu+3*Nx*Nx+2*Nx*Na)],
            'ua': [i for i in range(Nx+Nu+3*Nx*Nx+2*Nx*Na, Nx+Nu+3*Nx*Nx+2*Nx*Na+Nu*Na)]
        };
        return meta;

    x = X[:Nx].reshape(Nx,1);
    u = X[Nx:].reshape(Nu,1);
    a = aList;

    da = vec((x-a)@(x-a).T);
    xa = vec(x@a.T);
    ua = vec(x@a.T);    

    PsiX = obs(X);
    PsiA = np.vstack( (PsiX, da, xa, ua ) );

    return PsiA;


# plot results
def plotcomp(x1List, x2List, filename=None):
    fig, axs = plt.subplots();
    axs.plot(x1List[0], x1List[1], label='Model');
    axs.plot(x2List[0], x2List[1], linestyle='--', label='KCE');

    plt.title('$x_0 = [3, -1.4, 3, -10]^\intercal$');
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    axs.axis('equal');
    fig.tight_layout();
    plt.legend();
    plt.grid();
    
    if filename is None:
        plt.show();
    else:
        plt.savefig(filename, dpi=600);


# main executable section
if __name__ == "__main__":
    # simulation variables
    T = 10;  Nt = round(T/dt) + 1;
    tList = np.array( [ [i*dt for i in range(Nt)] ] );


    # generate the randomized control policy
    randControl = lambda x: np.random.rand(Nu,1);


    # generate training data for Kx
    N0 = 10;
    X0 = 10*np.random.rand(Nx,N0) - 5;

    xData, uData = data.generate_data(tList, model, X0,
        control=randControl, Nu=Nu);


    # construct training data from xData and uData
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


