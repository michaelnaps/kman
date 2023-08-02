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
dt = 0.001
Nx = 2

# Duffing model.
def model(x) -> np.array:
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
    P = 7
    if x is None:
        return {'Nk':P*Nx+1}
    N = x.shape[1]
    x1P = [ x[0]**i for i in range( 2,P+1 ) ]
    x2P = [ x[1]**i for i in range( 2,P+1 ) ]
    psi = np.vstack( (x, x1P, x2P, np.ones( (1,N) )) )
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
    T = 10;  Nt = round( T/dt ) + 1
    tList = np.array( [ [i*dt for i in range( Nt )] ] )
    xList, _ = generate_data( tList, model, X0 )

    # Construct X and Y sets.
    X = stack_data( xList[:,:-1], N0, Nx, Nt-1 )
    Y = stack_data( xList[:,1:], N0, Nx, Nt-1 )

    # Koopman operator variaables.
    kman = KoopmanOperator( obsx )
    print( 'K:\n ', kman.edmd( X, Y, X0 ) )

    # Plot comparison results.
    fig, axs = plt.subplots()
    trueSwm = Swarm2D( X0[:2], fig=fig, axs=axs, zorder=5,
        radius=0.10, tail_length=500 )
    # kmanSwm = Swarm2D( X0[:2], fig=fig, axs=axs, zorder=1,
    #     radius=0.15, color='indianred', tail_length=100 )
    trueSwm.draw()
    # kmanSwm.draw()

    plt.axis( [-6, 6, -6, 6] )
    plt.gca().set_aspect( 'equal', adjustable='box' )
    plt.show( block=0 )

    input( "Press ENTER to begin simulation..." )
    psi = obsx( X0 )
    for i, x in enumerate( xList.T ):
        psi = kman.K@psi

        # Update simulation based on sets.
        if i % 10 == 0:
            trueSwm.update( x.reshape( N0, Nx ).T[:2] )
            # kmanSwm.update( psi[:2] )

            # Pause sim. for visualization.
            plt.pause( 1e-3 )
    input( "Press ENTER to exit program..." )
