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
    print('Cascade EDMD Complete.');

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
    N0 = 2;
    X, Y, XU0 = createData(tList, N0, Nt);
    kList = learnOperators(X, Y, XU0);

    # print results
    for k in kList:
        print(k);

    # simulation variables
    sim_time = 10;
    N0n = 25;

    x0 = np.array( [[-12], [-17]] );
    u0 = np.zeros( (Nu,1) );
    xu0 = np.vstack( (x0+noise(delta,(Nx,1)), u0) );
    Psi0 = kList[-1].obsY(xu0);

    tList, xList, PsiList, uList, uTrueList = generateTrajectoryData(kList[-1], sim_time, x0, Psi0);

    # simulation options
    ans = input("\nStationary, animated, animated complete or trajectory results? [s/a/t/n] ");
    if ans == 'all':
        xvhc, kvhc = animatedResults(tList, xList, PsiList, rush=1);
        xvhc.axs.set_title('$\delta=%.2f, ' % delta + '\\varepsilon=%.2f$' % eps);

        figAnim = xvhc.fig;  axsAnim = xvhc.axs;
        figTraj, axsTraj = trajPlotting(tList, xList, PsiList, uList, uTrueList);
        figStat, axsStat = stationaryResults(kList[-1], sim_time, N0n);
        plt.show();

    else:
        while ans != 'n':
            if ans == 's':
                # test comparison results
                fig, axs = stationaryResults(kList[-1], sim_time, N0n);
                plt.show();
            elif ans == 'a':
                # simulation variables
                xvhc, kvhc = animatedResults(tList, xList, PsiList);
            elif ans == 't':
                fig, axs = trajPlotting(tList, xList, PsiList, uList, uTrueList);
                plt.show();
            ans = input("\nStationary, animated or trajectory simulation? [s/a/t/n] ");