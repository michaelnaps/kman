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
Nx = 3

# Grid bounds.
Ng = 3  # Density of grid.
xBounds = (-2, 2)
yBounds = (-2, 2)
alph = xBounds[1] - xBounds[0]
gamm = alph/Ng
beta = 1

# Duffing model function.
def model(x):
    alpha = 1
    delta = 1
    gamma = 0.05
    epsil = 0.20
    omega = 1.10
    dx = np.array( [
        x[1],
        alpha*x[0] - delta*x[0]**3 - gamma*x[1] - epsil*np.cos( omega*x[2] ),
        1
    ] )
    return x + dt*dx

# Grid-related functions.
def gridIndex(x):
    # Calculate indeces from bounds.
    i = int( (alph - (x[1] - xBounds[0]))/gamm )
    j = int( (x[0] - xBounds[0])/gamm )

    # Return indeces.
    return (i, j)

def gridMap(X, Nt):
    # Initialize grid matrix.
    p = round( alph/gamm )
    G = np.zeros( (Nt, p, p) )

    # Derive element locations.
    for k, x in enumerate( X.T ):
        G[k][gridIndex( x )] = 1

    return G

# Basis functions.
def bas(x):
    pass

# Main execution block.
if __name__ == '__main__':
    # Initial positions.
    A = 1.5
    N0 = 1
    # X0 = np.vstack( (
    #     np.linspace( -A, A, N0 ),
    #     np.zeros( (Nx-1, N0) )
    # ) )
    X0 = np.vstack( (
        2*A*np.random.rand( Nx-1, N0 ) - A,
        np.zeros( (Nx-2, N0) )
    ) )

    # Generate trajectory data.
    T = 5;  Nt = round( T/dt ) + 1
    tList = np.array( [ [i*dt for i in range( Nt )] ] )
    xList, _ = generate_data( tList, model, X0 )

    # Start simulation?
    ans = input("Press ENTER to begin simulation... ")
    if ans == 'n':
        exit()

    # Simulation step freq.
    dtmin = 0.1
    if dt < dtmin:
        n = round( dtmin/dt )
    else:
        n = 1

    # Simulation time frame.
    Tsim = 100;  Ntsim = round( Tsim/dt ) + 1
    tSim = np.array( [[i*dt for i in range( Ntsim )]] )

    # Simulation entities.
    fig, axs = plt.subplots()
    trueSwrm = Swarm2D( X0[:2], fig=fig, axs=axs, zorder=5,
        radius=0.001, tail_length=round( Ntsim/n ) )
    trueSwrm.draw()

    # Final adjustments and show plot.
    plt.axis( [-2, 2, -2, 2] )
    plt.gca().set_aspect( 'equal', adjustable='box' )
    plt.show( block=0 )

    # Simulation block.
    xSim, _ = generate_data( tSim, model, X0 )
    for i, x in enumerate( xSim.T ):
        # Every minimum time step.
        if i % n == 0:
            # Update frame.
            trueSwrm.update( x.reshape( N0, Nx ).T[:2] )
            # Pause sim. for visualization.
            plt.pause( 1e-3 )

    # Exit program.
    input( "Press ENTER to exit program..." )