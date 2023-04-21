from anchors import *

# Perform Coordinate EDMD
def learnOperators(X, Y, X0):
    # Ku block diagonal matrix function
    def Mu(K):
        m = Nu;
        p = obsX()['Nk'];
        q = 1;
        b = obsH()['Nk'];
        Kblock = cp.vstack( (
            cp.hstack( (np.eye(p), np.zeros( (p,b*q) )) ),
            cp.hstack( (np.zeros( (m,p) ), cp.kron( np.eye(q), K)) )
        ) );
        return Kblock;
    def npMu(K):
        m = Nu;
        p = obsX()['Nk'];
        q = 1;
        b = obsH()['Nk'];
        Kblock = np.vstack( (
            np.hstack( (np.eye(p), np.zeros( (p,b*q) )) ),
            np.hstack( (np.zeros( (m,p) ), np.kron( np.eye(q), K)) )
        ) );
        return Kblock;

    # initialize operator variables and solve
    NkX = obsXU()['Nk'];
    kuvar = KoopmanOperator(obsH, obsU);
    kxvar = KoopmanOperator(obsXUH, obsXU, M=Mu(kuvar.K));

    Psi1 = kxvar.liftData(X, X0, kxvar.obsX)[0];
    Psi2 = kxvar.liftData(Y, X0, kxvar.obsY)[0];

    print(kxvar.K.shape, Mu(kuvar.K).shape);
    print(Psi2.shape, Psi1.shape);

    dK = 1;
    while dK > 1e-3:
        # copy kvar
        kxcopy = kxvar.K;
        kucopy = kuvar.K;

        # Kx problem setup
        shapeX = kxvar.K.shape;
        Kx = cp.Variable( shapeX );
        objX = cp.Minimize( cp.sum_squares(Psi2 - Kx@Mu(kuvar.K)@Psi1) );
        prbX = cp.Problem( objX );

        # solve for Kx
        kxvar.err = prbX.solve();
        kxvar.K = np.array( Kx.value );
        print(kxvar);

        # Ku problem setup
        shapeU = kuvar.K.shape;
        Ku = cp.Variable( shapeU );
        objU = cp.Minimize( cp.sum_squares(Psi2 - kxvar.K@Mu(Ku)@Psi1) );
        prbU = cp.Problem( objU );

        # solve for Ku
        kuvar.err = prbU.solve();
        kuvar.K = np.array( Ku.value );
        print(kuvar);

        # calculate dK
        dK = np.linalg.norm( kxvar.K - kxcopy ) + np.linalg.norm( kuvar.K - kucopy );
        print('dK:', dK);

    # calculate cumulative operator
    Kf = np.array( kxvar.K )@np.array( npMu(kuvar.K) );
    kvar = KoopmanOperator(obsXUH, obsXU, K=Kf);

    klist = (kxvar, kuvar, kvar);
    return klist;

# main executable section
if __name__ == "__main__":
    # simulation variables
    T = 1;  Nt = round(T/dt) + 1;
    tList = np.array( [ [i*dt for i in range(Nt)] ] );

    # create data for learning operators
    N0 = 5;
    X, Y, XU0 = createData(tList, N0, Nt);
    klist = learnOperators(X, Y, XU0);

    # print results
    for k in klist:
        print(k);

    # check results in static/animated sims
    ans = input("\nStationary or animated sim? [s/a] ");
    if ans == 's':
        # test comparison results
        N0n = 25;
        fig, axs = stationaryResults(klist[-1], tList, N0n);
        plt.show();
    elif ans == 'a':
        # simulation variables
        x0 = 20*np.random.rand(Nx,1)-10;
        xvhc, kvhc = animatedResults(klist[-1], x0);
