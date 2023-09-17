import sys
from os.path import expanduser
sys.path.insert( 0, expanduser('~')+'/prog/geom' )

import argparse
from duffing import *
from GEOM.Vehicle2D import *

parser = argparse.ArgumentParser()
parser.add_argument( '--save' )
args = parser.parse_args()
save = bool( args.save )

# Main execution block.
if __name__ == '__main__':
    # Simulation length.
    T = 50.0;  dt = 1e-3
    Nt = round( T/dt ) + 1
    tList = np.array( [ [i*dt for i in range( Nt )] ] )

    # Simulation step freq.
    dts = 0.01
    if dt < dts:
        n = round( dts/dt )
    else:
        n = 1

    # Initial condtions.
    N0 = 9
    dX = 4*np.array( [
        [0, 1, 0, -1, 0, 1, 1, -1, -1],
        [0, 0, 1, 0, -1, 1, -1, -1, 1] ] )
    X0 = np.pi/2*np.hstack( (
        np.array( [
            [0, 1, 1, -1, -1],
            [0, 1, -1, 1, -1],
            [0 for i in range( 5 )] ] ),
        np.vstack( (
            np.random.rand( 2, 4 ),
            [0 for i in range( 4 )]
        ) ) ) )

    # Plot vehicles.
    Ntail = round( Nt/n ) + 1
    fig, axs = plt.subplots()
    axs.scatter( X0[0]+dX[0], X0[1]+dX[1],
        marker='o', edgecolor='k', facecolor='none' )
    swrm = Swarm2D( X0[:2]+dX, fig=fig, axs=axs,
        color='k', radius=0.01, tail_length=Ntail )
    swrm.setLineWidth( 1.5 ).draw()

    # Final adjustments and show plot.
    delta = 2
    plt.axis(
        np.array( [
            min(dX[0]-delta), max(dX[0]+delta),
            min(dX[1]-delta), max(dX[1]+delta) ] ) )
    plt.gca().set_aspect( 'equal', adjustable='box' )
    plt.show( block=0 )

    # Simulation loop.
    X = X0
    for i in range( Nt ):
        # X = model3( X, dt=dt )
        X = model3( X, c=[1,1,1,1,1,0], dt=dt )
        if i % n == 0:
            swrm.update( X[:2]+dX )
            # plt.pause( 1e-3 )
    input( "Press ENTER to exit program... " )

    if save:
        fig.savefig( expanduser('~')+'/prog/kman/.figures/.tat.png',
            dpi=800 )
