from anchors import *

filepath = '/home/michaelnaps/bu_research/koopman_operators_in_series/figures'

# perform Cascade EDMD
def learnOperators(X, Y, X0):
    # Ku block diagonal matrix function.
    def Tu(kvar):
        m = Nu
        p = obsX()['Nk']
        q = 1
        b = obsH()['Nk']
        Kblock = np.vstack( (
            np.hstack( (np.eye(p), np.zeros( (p,b*q) )) ),
            np.hstack( (np.zeros( (m,p) ), np.kron( np.eye(q), kvar[0].K )) )
        ) )
        return Kblock

    # Initialize operator variables and solve.
    kuvar = KoopmanOperator(obsH, obsU)
    kxvar = KoopmanOperator(obsXUH, obsXU, T=Tu( (kuvar,) ))

    Klist = (kxvar, kuvar)
    Tlist = (Tu, )
    Klist = cascade_edmd(Tlist, Klist, X, Y, X0)
    print('Cascade EDMD Complete.')

    # Form the cumulative operator.
    Kf = Klist[0].K @ Tu( Klist[1:] )
    kvar = KoopmanOperator(obsXUH, obsXU, K=Kf)
    kvar.resError(X[0], Y[0], X0)

    # Return the individual operators and cumulative.
    return kxvar, kuvar, kvar

# main executable section
if __name__ == "__main__":
    # simulation variables
    T = 1;  Nt = round(T/dt) + 1
    tList = np.array( [ [i*dt for i in range(Nt)] ] )

    # create data for learning operators
    N0 = 2
    xTrain, yTrain, XU0 = createData(tList, N0, Nt)
    X = (xTrain, xTrain);  Y = (yTrain, yTrain)
    kList = learnOperators(X, Y, XU0)

    # print results
    for k in kList:
        print(k)

    # simulation variables
    N0n = 15
    sim_time = 10

    x0 = np.array( [[-12], [-17]] )
    u0 = np.zeros( (Nu,1) )
    xu0 = np.vstack( (x0+[[-1.3],[1.5]], u0) )
    Psi0 = kList[-1].obsY.lift(xu0)
    tList, xList, PsiList, uList, uTrueList = generateTrajectoryData(kList[-1], sim_time, x0, Psi0)

    # eList = [i for i in range(0,11,2)]
    # pathComparisons(kList[-1], sim_time, x0, Psi0, eList)
    # plt.show()

    # simulation options
    ans = input("\nStationary, animated, animated complete or trajectory results? [s/a/t/all/n] ")
    if ans == 'all' or ans == 'save':
        xvhc, kvhc = animatedResults(tList, xList, PsiList, rush=1)
        xvhc.axs.set_title('$\delta=%.1f$, ' % delta + '$\\varepsilon=%.2f$' % epsilon)

        figAnim = xvhc.fig;  axsAnim = xvhc.axs
        figTraj, axsTraj = trajPlotting(tList, xList, PsiList, uList, uTrueList)
        figStat, axsStat = stationaryResults(kList[-1], sim_time, N0n)

        figAnim.set_figwidth(4)

        figAnim.set_figheight(5)
        figTraj.set_figheight(5)
        figStat.set_figheight(5)

        if ans == 'save':
            # pass
            figAnim.savefig(filepath+'/singlePathEnvironment', dpi=600)
            figTraj.savefig(filepath+'/singlePathTrajectories', dpi=600)
            # figStat.savefig(filepath+'/multiplePathEnvironment_e%.3f' % epsilon + '.png', dpi=600)
        else:
            plt.show()

    else:
        while ans != 'n':
            if ans == 's':
                # test comparison results
                fig, axs = stationaryResults(kList[-1], sim_time, N0n)
                plt.show()
            elif ans == 'a':
                # simulation variables
                xvhc, kvhc = animatedResults(tList, xList, PsiList)
            elif ans == 't':
                fig, axs = trajPlotting(tList, xList, PsiList, uList, uTrueList)
                plt.show()
            ans = input("\nStationary, animated or trajectory simulation? [s/a/t/n] ")