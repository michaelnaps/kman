from anchors import *

# main executable section
if __name__ == "__main__":
    # simulation variables
    T = 1;  Nt = round(T/dt) + 1;
    tList = np.array( [ [i*dt for i in range(Nt)] ] );

    # create data for learning operators
    N0 = 2;
    X, Y, XU0 = createData(tList, N0, Nt);

    kvar, kxvar, kuvar = learnOperators(X, Y, XU0);
    klist = (kxvar, kuvar, kvar);

    for k in klist:
        print(k);
