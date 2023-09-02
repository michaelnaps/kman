import sys
from os.path import expanduser
sys.path.insert( 0, expanduser('~')+'/prog/geom' )  # Plotting and sim classes.

from duffing import *
from GEOM.Vehicle2D import *

# Main execution block.
if __name__ == '__main__':
    # Simulation length.
    T = 1000;  dt = 0.001
    Nt = round( T/dt ) + 1
    tList = np.array( [ [i*dt for i in range( Nt )] ] )

    # Initial condtions.
    N0 = 1
    X0 = np.array( [[-1.5],[1.5],[0]] )

    # Plot vehicles.
    fig, axs = plt.subplots()
    swrm = Swarm2D( X0[:2], fig=fig, axs=axs, color='k',
        radius=0.05, tail_length=250 ).draw()

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

    # Simulation loop.
    X = X0
    for i in range( Nt ):
        X = model3( X, c=[1,1,1,1,1,0], dt=dt )
        if i % n == 0:
            swrm.update( X[:2] )
            plt.pause(1e-3)
