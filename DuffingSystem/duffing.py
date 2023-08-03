# Path imports.
import sys
from os.path import expanduser
sys.path.insert( 0, expanduser('~')+'/prog/kman' )
sys.path.insert( 0, expanduser('~')+'/prog/geom' )

# Standard imports.
import numpy as np
from KMAN.Operators import *
from GEOM.Vehicle2D import *

# Hyper parameter(s).
P = 3
dt = 0.001
Nx = 2

# Anchor initializations.
Na = 4
aList = 5*np.array( [
    [-1, 1, 0, 0],
    [0, 0, -1, 1]
] )

def anchorMeasure(x):
    da = np.empty( (Na,1) )
    for i, a in enumerate(aList.T):
        da[i] = np.linalg.norm(x - a[:,None])
    return da

# Duffing model.
def model(x):
    alpha = 5
    delta = 0.2
    gamma = 1
    dx = np.array( [
        x[1],
        alpha*x[0] - delta*x[0]**3 - gamma*x[1]
    ] )
    return x + dt*dx

# Observation spaces of interest.
def obsx(x=None):
    if x is None:
        return {'Nk':P*Nx+P*Na+1}
    d = anchorMeasure( x )
    x1P = [ x[0]**i for i in range( 2,P+1 ) ]
    x2P = [ x[1]**i for i in range( 2,P+1 ) ]
    dP = np.hstack( [d**i for i in range(1,P+1)] )
    dP = dP.reshape(P*Na,1)
    psi = np.vstack( (x, x1P, x2P, dP, [1]) )
    return psi

# Main execution block.
if __name__ == '__main__':
    # Initial positions.
    A = 2.5
    N0 = 10
    X0 = np.vstack( (
        np.linspace( -A, A, N0 ),
        np.zeros( (1, N0) )
    ) )

    # Generate trajectory data.
    T = 100;  Nt = round( T/dt ) + 1
    tList = np.array( [ [i*dt for i in range( Nt )] ] )
    xList, _ = generate_data( tList, model, X0 )

    # Construct X and Y sets.
    X = stack_data( xList[:,:-1], N0, Nx, Nt-1 )
    Y = stack_data( xList[:,1:], N0, Nx, Nt-1 )

    # Koopman operator variaables.
    kman = KoopmanOperator( obsx )
    print( 'K:\n ', kman.edmd( X, Y, X0 ) )

    # Define remeasurement function.
    def rmes(psi):
        x = psi[:Nx]
        d = anchorMeasure( x )
        psix = psi[:P*Nx]
        psid = np.hstack( [d**i for i in range(1,P+1)] ).reshape(P*Na,1)
        return np.vstack( (psix, psid, [1]) )

    # Plot comparison results.
    fig, axs = plt.subplots()
    anchors = [ Circle( a[:,None], fig=fig, axs=axs,
        radius=0.25, color='cornflowerblue' ).draw() for a in aList.T ]
    trueSwrm = Swarm2D( X0[:2], fig=fig, axs=axs, zorder=5,
        radius=0.10, tail_length=500 )
    kmanSwrm = Swarm2D( X0[:2], fig=fig, axs=axs, zorder=1,
        radius=0.15, color='indianred', tail_length=500 )
    trueSwrm.draw()
    kmanSwrm.draw()

    # Final adjustments and show plot.
    plt.axis( [-6, 6, -6, 6] )
    plt.gca().set_aspect( 'equal', adjustable='box' )
    plt.show( block=0 )

    # Run sim?
    ans = input( "Press ENTER to begin simulation... " )
    if ans == 'n':
        exit()

    # Simulation step freq.
    if dt < 0.01:
        n = round( 0.1/dt )
    else:
        n = 1

    # Simulation block.
    psi = np.hstack( ([obsx( x0[:,None] ) for x0 in X0.T]) )
    for i, x in enumerate( xList.T ):
        psi = kman.K@psi

        # Update simulation based on sets.
        if i % n == 0:
            trueSwrm.update( x.reshape( N0, Nx ).T[:2] )
            kmanSwrm.update( psi[:2] )

            # Pause sim. for visualization.
            plt.pause( 1e-3 )

    # Exit.
    input( "Press ENTER to exit program..." )
