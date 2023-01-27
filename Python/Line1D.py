import numpy as np
import matplotlib.pyplot as plt

from KoopmanSolve import *

def obsX(x):
    return x;

def obsU(x):
    return obsX(x);

def obsXU(X):
    x = X[0:2].reshape(2,1);
    u = X[2];

    PsiX = obsX(x);
    PsiU = obsU(x);

    return np.vstack( (PsiX, np.kron(PsiU, u)) );

def plot(tlist, xlist):
    fig, ax = plt.subplots();
    ax.plot(tlist, xlist[0]);
    ax.plot(tlist, xlist[1]);

if __name__ == "__main__":
    Nx = 2;
    Nu = 1;
    x0 = np.array( [[1],[1]] );
    u0 = np.array( [0] );

    # model equations
    xg = np.array( [[0],[0]] );
    dt = 0.1;
    A = np.array( [
        [1, dt],
        [0, 1]
    ] );
    B = np.array( [
        [0],
        [dt]
    ] );
    K = np.array( [
        [10, 5]
    ] );

    model = lambda x,u: A @ x.reshape(Nx,1) + B @ u.reshape(Nu,1);
    control = lambda x: K @ (xg.reshape(Nx,1) - x.reshape(Nx,1));

    # simulate model and control
    T = 5;
    Nt = round(T/dt) + 1;
    tlist = np.array([i*dt for i in range(Nt)]);

    ulist = np.zeros( (Nu, Nt) );
    xlist = np.zeros( (Nx, Nt) );

    ulist[:,0] = u0.reshape(Nu,);
    xlist[:,0] = x0.reshape(Nx,);

    for i in range(Nt-1):
        uNew = control(xlist[:,i]);
        ulist[:,i] = uNew.reshape(Nu,);

        xNew = model(xlist[:,i], ulist[:,i]);
        xlist[:,i+1] = xNew.reshape(Nx,);

    # plot(tlist, xlist);
    # plt.show()

    # create large data set
    N0 = 10;
    X0 = 10*np.random.rand( Nx, N0 ) - 5;
    xtrain = generate_data(model, tlist, X0, control, Nu);

    print(xtrain);
