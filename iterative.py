from anchors import *

# set global output setting
np.set_printoptions(precision=5, suppress=True);


# hyper paramter(s)
eps = 1;
delta = 0.1;
dt = 0.01;
Nx = 2;
Nu = 2;
Na = 3;
aList = np.array( [[10, 10, -10],[10, -10, -10]] );

# Na = 5;
# aList = np.array( [[10, 10, -10, -10, -5],[10, -10, -10, 10, -5]] );


# main executable section
if __name__ == "__main__":
    # simulation variables
    T = 1;  Nt = round(T/dt) + 1;
    tList = np.array( [ [i*dt for i in range(Nt)] ] );

    # create data for learning operators
    N0 = 2;
    X, Y, XU0 = createData(tList, N0, Nt);

    kxvar, kuvar, kvar = learnOperators(X, Y, XU0);
    klist = (kxvar, kuvar, kvar);

    for k in klist:
        print(k);


    ans = input("\nStationary or animated sim? [s/a] ");
    if ans == 's':
        # test comparison results
        N0n = 25;
        fig, axs = stationaryResults(kvar, tList, N0n);
        plt.show();
    elif ans == 'a':
        # simulation variables
        x0 = 20*np.random.rand(Nx,1)-10;
        xvhc, kvhc = animatedResults(kvar, x0);
