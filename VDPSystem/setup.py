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
dt = 0.01
Nx = 2

# Duffing model.
def model(x):
    mu = 1
    dx = np.array( [
        x[1],
        mu*(x[1] - x[1]*x[0]**2) - x[0]
    ] )
    return x + dt*dx

# Observation spaces of interest.
def obsx(x=None):
    if x is None:
        return {'Nk':Nx+1}
    N = x.shape[1]
    psi = np.vstack( (x, np.ones( (1,N) )) )
    return psi

# Main execution block.
if __name__ == '__main__':
    # Initial positions.
    A = 2.5
    N0 = 5
    X0 = 2*A*np.random.rand( Nx, N0 ) - A

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
    trueSwm = Swarm2D( X0, fig=fig, axs=axs,
        radius=0.10, zorder=10, tail_length=100 )
    kmanSwm = Swarm2D( X0, fig=fig, axs=axs,
        radius=0.15, color='indianred', tail_length=100 )
    kmanSwm.draw()
    trueSwm.draw()

    plt.axis( [-4, 4, -4, 4] )
    plt.gca().set_aspect( 'equal', adjustable='box' )
    plt.show( block=0 )

    psi = obsx( X0 )
    for x in xList.T:
        # Update true swarm based on set.
        trueSwm.update( x.reshape( N0, Nx ).T )

        # Update Koopman operator apprx.
        psi = kman.K@psi
        kmanSwm.update( psi[:Nx] )

        # Pause sim.
        plt.pause( 1e-3 )
