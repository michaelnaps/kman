# script for model equation testing
from diffdrive import *

def flow(u, x0):
    mvar.setObjectiveFunction( mvar.costFunctionGenerator( x0[:,None] ) )
    g = mvar.grad( u[:,None] )
    un = u[:,None] - alpha*g
    return un

def createDynamicSets(tList, X0):
    # model function for training syntax
    modelTrain = lambda x, u: model( x,u )
    controlTrain = lambda x: np.vstack( (np.random.rand(Nu,1), np.zeros( (Nu*(PH-1),1) )) )
    # np.array( mvar.solve(x, uinit)[0][:Nu] ).reshape(Nu,1)
    xTrain, uTrain = generate_data(tList, modelTrain, X0, controlTrain, Nu*PH)

    # split training data into X and Y sets
    uStack = stack_data(uTrain, N0, Nu*PH, Nt-1)
    xStack = stack_data(xTrain[:,:-1], N0, Nx, Nt-1)
    yStack = stack_data(xTrain[:,1:], N0, Nx, Nt-1)

    X = np.vstack( (xStack, uStack) )
    Y = np.vstack( (yStack, uStack) )

    # reshape initial condition set
    XU0 = np.vstack( (X0, np.zeros( (Nu, N0) )) )

    return X, Y, XU0

def createControlSets(iList, X0):
    # dimension variables
    NuPH = Nu*PH
    N0 = len( X0[0] )
    Ni = len( iList[0] )

    # use flow function at varying positions
    U0 = np.zeros( (NuPH, N0) )

    i = 0;  j = 0
    uTrain = np.empty( (N0*NuPH, Ni) )
    xTrain = np.empty( (N0*Nx, Ni-1) )
    for k in range(N0):
        control = lambda u: X0[:,k,None]  # position treated as constant 'control'
        uTemp, xTemp = generate_data( iList, flow, U0[:,k,None], control, Nx )

        uTrain[i:i+NuPH,:] = uTemp
        xTrain[j:j+Nx,:] = xTemp

        i = i + NuPH
        j = j + Nx

    # split training data into snapshots
    xStack  = stack_data( xTrain, N0, Nx, Ni-1 )
    u1Stack = stack_data( uTrain[:,:-1], N0, NuPH, Ni-1 )
    u2Stack = stack_data( uTrain[:,1:], N0, NuPH, Ni-1 )

    U1 = np.vstack( (u1Stack, xStack) )
    U2 = np.vstack( (u2Stack, xStack) )
    UX0 = np.vstack( (U0, X0) )

    return U1, U2, UX0

if __name__ == "__main__":
    # observable dimensions variables
    NkX = obsX()['Nk']  # for reference
    NkU = obsU()['Nk']
    NkH = obsH()['Nk']

    # simulation variables and data gen.
    T = 10;  Nt = round(T/dt)+1
    tList = np.array( [[i*dt for i in range(Nt)]] )


    # model functions
    # generate initial conditions for training
    A = 10
    N0 = 10
    X0 = 2*A*np.random.rand(Nx,N0) - A
    # X, Y, XU0 = createDynamicSets(tList, X0)

    # kxvar = KoopmanOperator(obsXUH, obsXU)
    # kxvar = kxvar.edmd(X, Y, XU0)

    # print('Kx:\n', kxvar)
    # print('Kx.PsiX:\n', kxvar.K[:NkX,:].T)
    # print('Kx.PsiU:\n', kxvar.K[NkX:,:].T)

    # # evaluate the behavior of Kx with remeasurement function
    # x0ref = np.array( x0 )[:,None]
    # uref = np.vstack( ([[1],[2]], np.zeros( (Nu*(PH-1),1) )) )
    # xTest, PsiTest = posTrackingNoControl(tList, kxvar, x0ref, uref)

    # # plot test results
    # plotcomp(tList, xTest, PsiTest)
    # plt.show()


    # control flow functions
    iList = np.array( [ [i for i in range( max_iter )] ] )
    U1, U2, UX0 = createControlSets( iList, X0 )

    kuvar = KoopmanOperator( obsUGX )
    kuvar.edmd( U1, U2, UX0 )

    print( 'Ku:\n', kuvar )
    print(  kuvar.K[:2*Nu,:].T )