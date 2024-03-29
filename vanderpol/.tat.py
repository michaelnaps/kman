import sys
from os.path import expanduser
sys.path.insert( 0, expanduser('~')+'/prog/geom' )  # Plotting and sim classes.

from GEOM.Vehicle2D import *


# Hyper parameter(s)
mu = 1


# Van der Pol oscilator.
def model(X, mu=1, dt=1e-3):
    N = X.shape[1]
    dX = np.empty( (2, N) )
    for i, x in enumerate( X.T ):
        dX[:,i] = np.array( [
            x[1],
            mu*(x[1] - x[1]*x[0]**2) - x[0]
        ] )
    return X + dt*dX


# Main execution block.
if __name__ == '__main__':
    # Simulation length.
    T = 50.0;  dt = 1e-3
    Nt = round( T/dt ) + 1
    tList = np.array( [ [i*dt for i in range( Nt )] ] )

    # Initial condtions.
    N0 = 9
    dX = mu*6*np.array( [
        [0, 1, 0, -1, 0, 1, 1, -1, -1],
        [0, 0, 1, 0, -1, 1, -1, -1, 1] ] )
    X0 = np.pi/2*np.hstack( (
        np.array( [
            [1e-6, 1, 1, -1, -1],
            [0, 1, -1, 1, -1]
        ] ),
        np.random.rand( 2, 4 )
    ) )

    # Plot vehicles.
    Ntail = round( Nt/25 )
    fig, axs = plt.subplots()
    axs.scatter( X0[0]+dX[0], X0[1]+dX[1], marker='o', edgecolor='k', facecolor='none' )
    swrm = Swarm2D( X0[:2]+dX, fig=fig, axs=axs, color='k', radius=0.01, tail_length=Ntail )
    swrm.setLineWidth( 1.5 ).draw()

    # Final adjustments and show plot.
    plt.axis( np.array( [min(dX[0]-3), max(dX[0]+3), min(dX[1]-3), max(dX[1]+3)] ) )
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
        # X = model3( X, dt=dt )
        X = model( X, mu=mu, dt=dt )
        if i % n == 0:
            swrm.update( X[:2]+dX )
            # plt.pause( 1e-3 )
    input( "Press ENTER to exit program... " )
