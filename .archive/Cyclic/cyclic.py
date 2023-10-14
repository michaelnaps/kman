import sys
sys.path.insert(0, '../')

from root import *

# Main execution block.
if __name__ == '__main__':
    # Dimensions variables.
    p = obsXUH()['p']
    q = obsXUH()['q']
    b = obsXUH()['b']

    # Initial conditions.
    x0 = np.array( [[R],[0],[0]] )
    xu0 = np.vstack( (x0, [[0],[0],[0]]) )

    # simulation time variables
    T = 10;  Nt = round( T/dt ) + 1
    tList = [ [i*dt for i in range( Nt )] ]

    # generate training data
    N0 = 1
    # X0 = np.vstack( (np.random.rand(Nx-2,N0), np.zeros( (Nx-1,N0) )) )
    xData, uData = generate_data( tList, model, x0, control=control, Nu=Nu )

    # formatting training data from xData and uData
    uStack = stack_data( uData, N0, Nu, Nt-1 )
    xStack = stack_data( xData[:,:-1], N0, Nx, Nt-1 )
    yStack = stack_data( xData[:,1:], N0, Nx, Nt-1 )

    XU0 = np.vstack( (x0, np.zeros( (Nu,N0) )) )
    X = np.vstack( (xStack, np.zeros( (Nu,N0*(Nt-1)) )) )
    Y = np.vstack( (yStack, uStack) )

    # learn Koopman operator
    def Tu(kuvar):
        m = Nu - 1
        Kblock = np.vstack(
            (np.hstack( (np.eye( p,p ), np.zeros( (p,q*b) )) ),
            np.hstack( (np.zeros( (q*m,p) ), np.kron( np.eye( q,q ), kuvar.K )) ))
        )
        return Kblock
    kuvar = KoopmanOperator( obsH, obsU )
    kxvar = KoopmanOperator( obsXUH, obsXU, T=Tu( kuvar ) )

    # Form list and compute results of c-EDMD.
    Klist = (kxvar, kuvar)
    Tlist = (Tu, )
    Klist = cascade_edmd( Klist, Tlist, X, Y, x0 )

    # Form the cumulative operator.
    Kf = Klist[0].K @ Tu( Klist[1] )
    kvar = KoopmanOperator( obsXUH, obsXU, K=Kf )
    kvar.resError( X, Y, x0 )
    print( kvar )

    # Simulate model using re-measurement function.
    def rmes( Psi ):
        x = Psi[:Nx]
        d = anchorMeasure( x )
        PsiX = Psi[:p]
        PsiU = np.array( [[1]] )
        PsiH = d**2
        PsiXUH = np.vstack( (PsiX, np.kron( PsiU, PsiH )) )
        return PsiXUH
    kModel = lambda Psi, u: kvar.K@rmes( Psi )
    vhc, xList = simulateModelWithControl( obsXU( xu0 ), kModel, N=10000 )

    # # plot static results
    # fig, axs = plt.subplots()
    # axs.plot( xList[:Nx-1,:].T )
    # plt.show()
