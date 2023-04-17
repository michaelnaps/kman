from anchors import *

# closed-loop observation functions
def obs(X=None):
    if X is None:
        meta = {'Nk': 1};
        return meta;
    Psi = X;
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
