import numpy as np
import matplotlib.pyplot as plt
import Helpers.DataFunctions as dfun
import Helpers.KoopmanFunctions as kman


def observables(X=None):
    if X is None:
        meta = {
            'Nk': 6,
            'x': [1,2,3,4],
            'u': [5,6]
        };
        return meta;

    Psi = X;

    return Psi;

def plot(xlist, ylist):
    fig, axs = plt.subplots(2, 1);
    axs[0].plot(xlist);
    axs[1].plot(ylist);
    plt.show();

    return fig, axs;


if __name__ == "__main__":
    # model dimension parameters
    Nx = 4;
    Nu = 2;

    # model parameters
    dt = 0.1;
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
    C = np.array( [
        [10, 0, 2.5, 0],
        [0, 10, 0, 2.5]
    ] );

    model = lambda x,u: A@x.reshape(Nx,1) + B@u.reshape(Nu,1);

    xg = np.zeros( (Nx,1) );
    control = lambda x: C@(xg.reshape(Nx,1) - x.reshape(Nx,1));
    random_control = lambda x: 2*np.random.rand(Nu,1) - 1;

    # generate model data
    N0 = 10;
    X0 = np.random.rand(Nx,N0);

    T = 10;  Nt = round(T/dt) + 1;
    tTrain = np.array( [i*dt for i in range(Nt)] );

    xData, uData = dfun.generate_data(tTrain, model, X0, control=control, Nu=Nu)

    X = np.vstack( (xData[:,:Nt-1], uData) );
    Y = np.vstack( (xData[:,1:Nt], uData) );

    Xstack = dfun.stack_data(X, N0, Nx+Nu, Nt-1);
    Ystack = dfun.stack_data(Y, N0, Nx+Nu, Nt-1);

    plot(Xstack, Ystack)

    print('X', Xstack.shape);
    print(Xstack)

    print('Y', Ystack.shape);
    print(Ystack);

    # solve for Koopman operator
    kvar = kman.KoopmanOperator(observables);
    K = kvar.edmd(Xstack, Ystack, np.vstack( (X0, np.zeros( (Nu,N0) )) ));

    print('K:\n', K);
