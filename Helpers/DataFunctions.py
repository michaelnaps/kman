import numpy as np

def generate_data(tlist, model, X0, control=None, Nu=0):
    Nx = len(X0);
    N0 = len(X0[0]);
    Nt = len(tlist);

    if control is not None:
        ulist = np.zeros( (N0*Nu, Nt-1) );
    else:
        Nu = Nx;

    xlist = np.zeros( (N0*Nx, Nt) );

    n = 0;
    k = 0;
    for i in range(N0):

        u = np.zeros( (Nu, Nt-1) );
        x = np.zeros( (Nx, Nt) );

        x[:,0] = X0[:,i];

        for t in range(Nt-1):
            if control is not None:
                u[:,t] = control(x[:Nx,t]).reshape(Nu,);
                x[:,t+1] = model(x[:Nx,t], u[:,t]).reshape(Nx,);

        ulist[k:k+Nu,:] = u;
        xlist[n:n+Nx,:] = x;
        n = n + Nx;
        k = k + Nu;

    return xlist, ulist;


def stack_data(data, N0, Nx, Nt):
    x = np.zeros( (Nx, N0*Nt) );

    k = 0;
    t = 0;
    for i in range(N0):
        x[:,t:t+Nt] = data[k:k+Nx,:];
        k += Nx;
        t += Nt;

    return x;
