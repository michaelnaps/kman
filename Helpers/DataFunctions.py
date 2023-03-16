import numpy as np
import matplotlib.pyplot as plt


def generate_data(tlist, model, X0, control=None, Nu=0):
    Nx = len(X0);
    N0 = len(X0[0]);
    Nt = len(tlist[0]);

    if control is not None:
        ulist = np.empty( (N0*Nu, Nt-1) );
    else:
        ulist = None;
        Nu = Nx;

    xlist = np.empty( (N0*Nx, Nt) );

    n = 0;
    k = 0;
    for i in range(N0):

        if control is not None:
            u = np.empty( (Nu, Nt-1) );
        x = np.empty( (Nx, Nt) );

        x[:,0] = X0[:,i];

        for t in range(Nt-1):
            if control is not None:
                u[:,t] = control(x[:Nx,t]).reshape(Nu,);
                x[:,t+1] = model(x[:Nx,t], u[:,t]).reshape(Nx,);
            else:
                x[:,t+1] = model(x[:Nx,t]).reshape(Nx,);

        if control is not None:
            ulist[k:k+Nu,:] = u;
            k = k + Nu;

        xlist[n:n+Nx,:] = x;
        n = n + Nx;

    return xlist, ulist;


def stack_data(data, N0, Nx, Nt):
    x = np.empty( (Nx, N0*Nt) );

    k = 0;
    t = 0;
    for i in range(N0):
        x[:,t:t+Nt] = data[k:k+Nx,:];
        k += Nx;
        t += Nt;

    return x;


def compare_data(Xlist, Ylist, X0, fig=None, axs=None):
    # plot test results
    if fig is None:
        fig, axs = plt.subplots();

    n, m = X0.shape;
    axs.plot(Xlist[-n], Xlist[-(n+1)], color='b', label='Model');
    axs.plot(Ylist[-n], Ylist[-(n+1)], color='r', linestyle='--', label='KCE');

    k = 0;
    for x0 in X0.T[:-n]:
        axs.plot(Xlist[k], Xlist[k+1], color='b');
        axs.plot(Ylist[k], Ylist[k+1], color='r', linestyle='--');

        axs.axis( (-10,10,-10,10), aspect='equal' );
        axs.legend();
        axs.grid();

        k += n;

    return fig, axs;
