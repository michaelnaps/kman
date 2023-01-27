import numpy as np
import matplotlib.pyplot as plt

from KoopmanFunctions import *


def obsX(x=None):
    if x is None:
        Nk = 2;
        return Nk;

    PsiX = x;
    return PsiX;

def obsU(x=None):
    if x is None:
        Nk = 1;
        return Nk;

    PsiU = [[1]];
    return PsiU;

def obsXU(X=None):
    if X is None:
        Nk = obsX() + obsU()*obsH();
        return Nk;

    x = X[0:2].reshape(2,1);
    u = X[2];

    PsiX = obsX(x);
    PsiU = obsU(x);
    PsiH = obsH(X);

    PsiXU = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );

    return PsiXU;

def obsH(X=None):
    if X is None:
        Nk = 3;
        return Nk;

    PsiH = X;
    return PsiH;


def plot(tlist, xlist):
    fig, ax = plt.subplots();
    ax.plot(tlist, xlist[0]);
    ax.plot(tlist, xlist[1]);


if __name__ == "__main__":
    Nx = 2;
    Nu = 1;
    x0 = np.array( [[1],[1]] );
    u0 = np.array( [[0]] );

    # model equations
    xg = np.array( [[0],[0]] );
    dt = 0.1;
    A = np.array( [[1, dt], [0, 1]] );
    B = np.array( [[0], [dt]] );
    K = np.array( [[10, 5]] );

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
        ulist[:,i+1] = uNew.reshape(Nu,);

        xNew = model(xlist[:,i], ulist[:,i+1]);
        xlist[:,i+1] = xNew.reshape(Nx,);


    # solve for Ku
    xu0 = np.vstack( (x0, u0) );
    Xu = np.vstack( (xlist, np.zeros( (Nu, Nt) )) );
    Yu = np.vstack( (xlist, ulist) )

    NkH = obsH();
    Ku, _, _ = KoopmanSolve(obsH, NkH, Xu, Yu, xu0)

    print(Ku);


    # create large data set for Ku and Kx solution
    N0 = 10;
    X0 = 10*np.random.rand( Nx, N0 ) - 5;
    xtrain, utrain = generate_data(tlist, model, X0, control, u0);
    xtrain = stack_data(xtrain, N0, Nx, Nt);
    utrain = stack_data(utrain, N0, Nu, Nt);


    # solve for Kx
    Xx = np.vstack( (xtrain[:,:Nt-1], utrain[:,:Nt-1]) );
    Yx = np.vstack( (xtrain[:,1:Nt],  utrain[:,1:Nt]) );

    NkXU = obsXU();
    Kx, err, ind = KoopmanSolve(obsXU, NkXU, Xx, Yx, xu0);

    print(err);
    print(ind);
    print(Kx);


    # generate the cumulative operator
    # K = Kx*[I in (p x p), 0 in (p x mq); 0 in (mq x p), kron(Ku, I in q)]
    m = Nu;
    p = obsX();
    q = obsU();
    b = NkH;

    top = np.hstack( (np.eye(p), np.zeros( (p, b*q) )) );
    bot = np.hstack( (np.zeros( (b*q, p) ), np.kron(Ku, np.eye(q))) );

    K = Kx @ np.vstack( (top, bot) );

    print(K)
