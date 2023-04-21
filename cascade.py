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

    klist = (kxvar, kuvar);
    mlist = (Mu, );
    klist = cascade_edmd(klist, mlist, X, Y, X0);

    # form the cumulative operator
    Kf = klist[0].K @ Mu( klist[1] );
    kvar = KoopmanOperator(obsXUH, obsXU, K=Kf);
    kvar.resError(X, Y, X0);

    return kxvar, kuvar, kvar;

# main executable section
if __name__ == "__main__":
    # simulation variables
    T = 5;  Nt = round(T/dt) + 1;
    tList = [ [i*dt for i in range(Nt)] ];

    # create data for learning operators
    N0 = 1;
    X, Y, XU0 = createData(tList, N0, Nt);
    klist = learnOperators(X, Y, XU0);

    # print results
    for k in klist:
        print(k);

    # simulation options
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