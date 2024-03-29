
from duffing import *

# Hyper parameters.
dt = 0.001
Nx = 3
Nf = 15  # Fourier expansion number.

# Ideal control.
C = -np.array( [[1, 0],[0, 1]] )
print( 'C\n', C )

# Model function.
c = [1,1,1,1,1,0]
model = lambda X, U=None: model3(X, U=U, c=c, dt=dt )

# Observation functions.
def obs1(X=None):
    if X is None:
        return {'Nk': Nx-1}
    psi1 = X[:Nx-1]
    return psi1

def obs2(X=None):
    if X is None:
        return {'Nk': 2}
    psi3 = np.vstack( (np.sin( X[2] ), np.cos( X[2] )) )
    return psi3

def obs12(X=None):
    if X is None:
        return {'Nk': obs1()['Nk']+obs2()['Nk']}
    psi1 = obs1( X )
    psi2 = obs2( X )
    psi12 = np.vstack( (psi1, psi2) )
    return psi12

def obs3(X=None):
    if X is None:
        return {'Nk': 2*Nf+1}
    xSin = [ np.sin( k*X[0] ) for k in range( 1,Nf+1 ) ]
    xCos = [ np.cos( k*X[0] ) for k in range( Nf+1 ) ]
    psi3 = np.vstack( (xSin, xCos) )
    return psi3

def obs3p(X=None):
    if X is None:
        return {'Nk': 1}
    x1 = X[0,None]
    psi3p = x1**3
    return psi3p

def obs123(X=None):
    if X is None:
        return {'Nk': obs1()['Nk']+obs2()['Nk']+obs3()['Nk']}
    psi1 = obs1( X )
    psi2 = obs2( X )
    psi3 = obs3( X )
    psi123 = np.vstack( (psi1, psi2, psi3) )
    return psi123

def obs123p(X=None):
    if X is None:
        return {'Nk': obs1()['Nk']+obs2()['Nk']+obs3p()['Nk']}
    psi1 = obs1( X )
    psi2 = obs2( X )
    psi3p = obs3p( X )
    psi123p = np.vstack( (psi1, psi2, psi3p) )
    return psi123p

# Main execution block.
if __name__ == '__main__':
    # Initialize time-series data.
    T = 1;  Ntt = round( T/dt ) + 1
    tTrain = np.array( [ [i*dt for i in range( Ntt )] ] )

    # State initialization for training.
    A = 3.0
    N0t = 10
    X0t = np.vstack( (
        2*A*np.random.rand( Nx-1, N0t ) - A,  # position init
        np.zeros( (Nx-2, N0t ) )              # time-series init
    ) )

    # Generate state data sets.
    xData = generate_data( tTrain, model, X0t )[0]
    xTrain = stack_data( xData[:,:-1], N0t, Nx, Ntt-1 )
    yTrain = stack_data( xData[:,1:], N0t, Nx, Ntt-1 )

    # Cascade sets.
    X = (xTrain, xTrain, xTrain)
    Y = (yTrain, yTrain, xTrain)

    # Initialize shift functions.
    p1 = obs1()['Nk'];   q1 = obs1()['Nk']
    p2 = obs2()['Nk'];   q2 = obs2()['Nk']
    p3 = obs3p()['Nk'];  q3 = obs3()['Nk']
    def shift( Klist ):
        T = np.eye( p1+p2+p3, q1+q2+q3 )
        T[p1:p1+p2,q1:q1+q2] = Klist[0].K
        T[p1+p2:,q1+q2:] = Klist[1].K
        return T

    # Initialize operator variables and solve.
    k3var = KoopmanOperator( obs3, obs3p )
    k2var = KoopmanOperator( obs2 )
    k1var = KoopmanOperator( obs123, T=shift( (k2var, k3var) ) )

    Klist = (k1var, k2var, k3var)
    Tlist = (shift, None)
    Klist = cascade_edmd( Tlist, Klist, X, Y, X0t )

    print( 'Cascade EDMD Complete.' )
    for K in Klist:
        print( K )

    # Calculate cumulative operator.
    kvar = KoopmanOperator( obs123 )
    kvar.setOperator( Klist[0].K@shift( Klist[1:] ) )
    # kvar.resError( xTrain, yTrain, save=1 )
    print( '\nKoopman operator:\n', kvar )

    # Calculate Lie operator.
    lvar = LieOperator( obs123 ).K2L( kvar, dt=dt )
    print( '\nLie operator:\n', lvar )

    # Start simulation?
    ans = input("\n\nPress ENTER to begin simulation... ")
    if ans == 'n':
        exit()

    # Simulate results.
    def rmes(PSI):
        N = PSI.shape[1]
        PSIn = np.empty( (obs123()['Nk'], N) )
        for i, psi in enumerate( PSI.T ):
            x = psi[:Nx-1,None]
            psi3 = obs3( x )
            PSIn[:,i] = np.vstack( (psi[:,None], psi3) )[:,0]
        return PSIn

    # Position Initializations.
    As = 1.5
    N0s = 10
    Xs = 2*As*np.random.rand( Nx, N0s ) - As
    PSIs = obs12( Xs )

    print( 'Initial States:\n', Xs )

    # Vehicle/plot initializations.
    R = 0.10;  Ntail = 150
    fig, axs = plt.subplots()
    trueSwrm = Swarm2D( Xs[:2], fig=fig, axs=axs, zorder=1,
        radius=R/2, tail_length=Ntail ).draw()
    kmanSwrm = Swarm2D( PSIs[:2], fig=fig, axs=axs, zorder=5,
        radius=R, color='cornflowerblue',
        tail_length=Ntail ).setLineStyle( ':' ).draw()

    # Final adjustments and show plot.
    plt.axis( [-2, 2, -2, 2] )
    plt.gca().set_aspect( 'equal', adjustable='box' )
    plt.show( block=0 )

    # Simulation step freq.
    dts = 0.1
    if dt < dts:
        n = round( dts/dt )
    else:
        n = 1

    # Simulation time parameters.
    T = 100;  Nt = round( T/dt ) + 1
    for i in range( Nt ):
        # Get new positions.
        PSIs = kvar.K[:obs12()['Nk']]@obs123( Xs )
        dPSIs = lvar.L[:obs12()['Nk']]@obs123( Xs )
        Xs = model( Xs, C@PSIs[:2] - dPSIs[:2] )

        # Update plots for appropriate time-steps.
        if i % n == 0:
            trueSwrm.update( Xs[:2] )
            kmanSwrm.update( PSIs[:2] )
            plt.pause( 1e-3 )

    # Exit program.
    input( "Press ENTER to exit program..." )