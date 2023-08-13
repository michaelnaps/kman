import sys
from os.path import expanduser
sys.path.insert( 0, expanduser('~')+'/prog/kman' )
sys.path.insert( 0, expanduser('~')+'/prog/four' )
sys.path.insert( 0, expanduser('~')+'/prog/geom' )

# Standard imports.
import numpy as np
from KMAN.Operators import *
from FOUR.Transforms import *
from GEOM.Vehicle2D import *
from GEOM.Polygon import *

# Hyper parameter(s).
dt = 0.001
Nx = 3

# Grid bounds.
Ng = 10  # Density of grid.
xBounds = (-2, 2)
yBounds = (-2, 2)
alph = xBounds[1] - xBounds[0]
gamm = alph/Ng
beta = 1

# Duffing model function.
def model(x):
    c = [ 1, 1, 0.07, 0.20, 1.10 ]
    dx = np.array( [
        x[1],
        c[0]*x[0] - c[1]*x[0]**3 - c[2]*x[1] - c[3]*np.cos( c[4]*x[2] ),
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

    # Return grip map.
    return G

def gridStack(gmap):
    # Stack grid.
    Nt = gmap.shape[0]
    N = round( alph/gamm )**2

    # Main execution loop.
    gstack = np.empty( (N, Nt) )
    for k, g in enumerate( gmap ):
        gstack[:,k] = g.reshape( N, )

    # Return grid stack.
    return gstack

def gridRemap(gstack):
    # Dimensions
    n = round( alph/gamm )
    Nt = gstack.shape[1]

    # Reform grid map.
    gmap = np.empty( (Nt, n, n) )
    for k, g in enumerate( gstack.T ):
        gmap[k] = g.reshape( n, n )

    # Return map.
    return gmap

# Main execution block.
if __name__ == '__main__':
    # Initial positions.
    A = 1.5
    N0 = 1
    X0 = np.vstack( (
        2*A*np.random.rand( Nx-1, N0 ) - A,
        np.zeros( (Nx-2, N0) )
    ) )

    # Generate trajectory data.
    T = 100;  Nt = round( T/dt ) + 1
    tList = np.array( [ [i*dt for i in range( Nt )] ] )
    xList, _ = generate_data( tList, model, X0 )
    gData = gridMap( xList, Nt )

    # Fourier series approximation.
    gTrain = gridStack( gData )
    fvar = RealFourier( tList, gTrain )
    fvar.dmd( N=100 )

    # Simulation step freq.
    dtmin = 0.1
    if dt < dtmin:
        n = round( dtmin/dt )
    else:
        n = 1

    # Simulation entities.
    fig, axs = plt.subplots()
    gridvar = Grid( gamm, xBounds, yBounds,
        fig=fig, axs=axs, color='grey', zorder=1 )
    trueSwrm = Swarm2D( X0[:2], fig=fig, axs=axs, zorder=5,
        radius=0.001, tail_length=round( Nt/n ) )
    gridvar.draw()
    trueSwrm.draw()

    # Final adjustments and show plot.
    plt.axis( [-2, 2, -2, 2] )
    plt.gca().set_aspect( 'equal', adjustable='box' )
    plt.show( block=0 )

    # Start simulation?
    ans = input("Press ENTER to begin simulation... ")
    if ans == 'n':
        exit()

    # Simulation block.
    gFour = fvar.solve( tList )
    gSim = gridRemap( gFour )
    for t, x, g in zip( tList.T, xList.T, gSim ):
        # Every minimum time step.
        if round( t[0]/dt ) % n == 0:
            for i in range( g.shape[0] ):
                for j in range( g.shape[1] ):
                    if g[i,j] > 1:
                        a = 1
                    elif g[i,j] < 0:
                        a = 0
                    else:
                        a = g[i,j]
                    color = (0.5, 0.5, 0.5, a)
                    gridvar.setCellColor( i, j, color )
            gridvar.update()
            trueSwrm.update( x.reshape( N0, Nx ).T[:2] )
            plt.pause( 1e-3 )

    # Exit program.
    input( "Press ENTER to exit program..." )