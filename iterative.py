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

# iterative learning procedure
def coordinateTesting(X, Y, X0):
    # Ku block diagonal matrix function
    def Mu(kvar):
        m = Nu;
        p = obsX()['Nk'];
        q = 1;
        b = obsH()['Nk'];
        Kblock = np.vstack( (
            np.hstack( (np.eye(p), np.zeros( (p,b*q) )) ),
            np.hstack( (np.zeros( (m,p) ), np.kron( np.eye(q), kvar.K)) )
        ) );
        return Kblock;

    # initialize operator variables and solve
    kuvar = KoopmanOperator(obsH, obsU);
    kxvar = KoopmanOperator(obsXUH, obsXU, M=Mu(kuvar));

    Psi1 = kxvar.liftData(X, X0, kxvar.obsX)[0];
    Psi2 = kxvar.liftData(Y, X0, kxvar.obsY)[0];

    print(kxvar.K.shape, Mu(kuvar).shape);
    print(Psi1.shape, Psi2.shape);

    dK = 1;
    while dK > 1e-3:
        # Kx section
        Kx = cp.Variable( kxvar.K.shape );
        objX = cp.Minimize( cp.sum_squares(Kx@Mu(kuvar)@Psi1 - Psi2) );
        prbX = cp.Problem(objX)

        # Ku section

        # compute overall change
        pass;

    return kxvar, kuvar, kvar;

# main executable section
if __name__ == "__main__":
    # simulation variables
    T = 1;  Nt = round(T/dt) + 1;
    tList = np.array( [ [i*dt for i in range(Nt)] ] );

    # create data for learning operators
    N0 = 1;
    X, Y, XU0 = createData(tList, N0, Nt);

    kxvar, kuvar, kvar = coordinateTesting(X, Y, XU0);
    klist = (kxvar, kuvar, kvar);

    for k in klist:
        print(k);

    # check results in static/animated sims
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
