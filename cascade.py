from anchors import *

# perform Cascade EDMD
def learnOperators(X, Y, X0):
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

    kList = (kxvar, kuvar);
    mlist = (Mu, );
    kList = cascade_edmd(kList, mlist, X, Y, X0);

    # form the cumulative operator
    Kf = kList[0].K @ Mu( kList[1] );
    kvar = KoopmanOperator(obsXUH, obsXU, K=Kf);
    kvar.resError(X, Y, X0);

    return kxvar, kuvar, kvar;

# main executable section
if __name__ == "__main__":
    # simulation variables
    T = 1;  Nt = round(T/dt) + 1;
    tList = [ [i*dt for i in range(Nt)] ];

    # create data for learning operators
    N0 = 1
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