from anchors import *

# closed-loop observation functions
def obs(X=None):
    if X is None:
        meta = {'Nk': Nx+Nu+3*Na+2};
        return meta;

    x = X[:Nx];
    u = X[Nx:];

    xTx = x.T@x;
    xTu = x.T@u;

    d = np.empty( (Na,1) );
    xTa = np.empty( (Na,1) );
    uTa = np.empty( (Na,1) );

    for i, a in enumerate(aList.T):
        d[i] = (x - a[:,None]).T@(x - a[:,None]);
        xTa[i] = x.T@a;
        uTa[i] = u.T@a;

    Psi = np.vstack( (x, u, d, xTx, xTu, xTa, uTa) );

    return Psi;

# main execution block
if __name__ == '__main__':
    # simulation data (for training)
    T = 10;  Nt = round(T/dt)+1;
    tList = [[i*dt for i in range(Nt)]];

    # generate data
    N0 = 10;
    X, Y, X0 = createData(tList, N0, Nt);

    # initialize operator
    kvar = KoopmanOperator( obs );
    kvar.edmd(X, Y, X0);
