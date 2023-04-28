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

    # print(kxvar.K.shape, Mu(kuvar.K).shape);
    # print(Psi2.shape, Psi1.shape);

    dK = 1;
    while dK > 1e-6:
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
        # print(kxvar);

        # Ku problem setup
        shapeU = kuvar.K.shape;
        Ku = cp.Variable( shapeU );
        objU = cp.Minimize( cp.sum_squares(Psi2 - kxvar.K@Mu(Ku)@Psi1) );
        prbU = cp.Problem( objU );

        # solve for Ku
        kuvar.err = prbU.solve();
        kuvar.K = np.array( Ku.value );
        # print(kuvar);

        # calculate dK
        dK = np.linalg.norm( kxvar.K - kxcopy ) + np.linalg.norm( kuvar.K - kucopy )**2;
        # print('dK:', dK);
        # print('------------');

    # calculate cumulative operator
    Kf = np.array( kxvar.K )@np.array( npMu(kuvar.K) );
    kvar = KoopmanOperator(obsXUH, obsXU, K=Kf);
    # print('Coordinate EDMD Complete.');

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
    kList = learnOperators(X, Y, XU0);

    # print results
    for k in kList:
        print(k);

    # simulation options
    sim_time = 5;
    ans = input("\nStationary, animated or trajectory simulation? [s/a/t] ");
    if ans == 's':
        # test comparison results
        N0n = 25;
        fig, axs = stationaryResults(kList[-1], sim_time, N0n);
        plt.show();
    elif ans == 'a':
        # simulation variables
        x0 = np.array( [[-12], [17]] )
        xvhc, kvhc = animatedResults(kList[-1], sim_time, x0);
    elif ans == 't':
        x0 = np.array( [[-12], [17]] )
        tList = [ [i*dt for i in range( round(sim_time/dt+1) )] ];
        fig, axs = trajPlotting(kList[-1], sim_time, x0);
        plt.show();
