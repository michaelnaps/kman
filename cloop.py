from anchors import *

# closed-loop observation functions
def obs(X=None):
    if X is None:
        meta = {'Nk': 1};
        return meta;

    d = np.empty( (Na,1) );
    xTx = np.empty( (Nx,1) );
    xTu = np.empty( (Nu,1) );

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
    kvar = KoopmanOperator(obs);
