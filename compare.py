from anchors import *
import cascade as casc
import coordinate as cord

# main execution block
if __name__ == '__main__':
    # simulation variables
    T = 10;  Nt = round(T/dt) + 1;
    tList = [ [i*dt for i in range(Nt)] ];

    # create data for learning operators
    N0 = 5;
    X, Y, XU0 = createData(tList, N0, Nt);

    # perform cascade and coordinate EDMD methods
    kCasc = casc.learnOperators(X, Y, XU0);
    kCord = cord.learnOperators(X, Y, XU0);

    # plot comparisons