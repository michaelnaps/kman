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
def obs(X=None):
    if X is None:
        meta = {'Nk':Nx+Nu};
        return meta;
    return X;


# main executable section
if __name__ == "__main__":
    # simulation variables
    T = 10;  Nt = round(T/dt) + 1;
    tList = np.array( [ [i*dt for i in range(Nt)] ] );


    # generate the randomized control policy
    randControl = lambda x: np.random.rand(Nu,1);


    # generate training data for Kx
    N0 = 1;
    X0 = 20*np.random.rand(Nx,N0) - 10;


    # construct training data from xData and uData
    xData, uData = data.generate_data(tList, model, X0,
        control=randControl, Nu=Nu);

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

    kuvar = kman.KoopmanOperator(obs);
    Ku = kuvar.edmd(Xu, Yu, XU0)

    print('Ku:', Ku.shape, kuvar.err)
    print(Ku);
    print('\n');


    # generate cumulative operator
    Nk = obs()['Nk'];
    K = Kx @ Ku;
    print('K\n', K, '\n')


    # new operator model equation
    kModel = lambda Psi: K@Psi;


    # data for testing results
    N0n = 25;
    X0n = 20*np.random.rand(Nx,N0n) - 10;
    XU0n = np.vstack( (X0n, np.zeros( (Nu,N0n) )) );
    
    Psi0 = np.empty( (Nk,N0n) );
    for i, xu in enumerate(XU0n.T):
        Psi0[:,i] = obs( xu.reshape(Nx+Nu,1) ).reshape(Nk,);

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
    