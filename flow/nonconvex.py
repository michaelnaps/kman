# Path imports.
import sys
from os.path import expanduser
sys.path.insert( 0, expanduser('~')+'/prog/kman' )  # Koopman operator classes.
sys.path.insert( 0, expanduser('~')+'/prog/mpc' )   # Optimization classes.

# Standard imports.
import numpy as np
import matplotlib.pyplot as plt

# Homemade imports.
from KMAN.Operators import *
from MPC.Optimizer import *

# Set global number print setting.
np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)

# Dimension of system.
n = 2
alpha = 1e-3
gamma = alpha
beta = 100

# Convex objective function.
def cost(x):
    g = (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    return g[:,None].T

def costgrad(x):
    dg = fdm2c( cost, x )
    return dg

def costgradprop(x, p=0):
    if p < 0:
        return None
    elif p != 0:
        dg = costgrad( x )
        return costgradprop( x - alpha*costgrad( x ), p=p-1 )
    return costgrad( x )

# Observation function.
def observe(x=None):
    p = 2
    if x is None:
        return {'Nk': (p + 1)*n + 1}
    psi = np.vstack( [x]
        + [gamma**(k+1)*costgradprop( x, p=k ) for k in range( p )]
        + [1] )
    return psi

# Main execution block.
if __name__ == '__main__':
    # Optimization variable.
    eps = 1e-21
    optvar = Optimizer( cost, eps=eps )
    optvar.setStepSize( alpha ).setMaxIter( np.inf )

    # Initial guess and system size.
    p = 2
    A = 2.5
    X0 = 2*A*np.random.rand( p,n,1 ) - A

    # Solve optimization problem and save steps.
    kmax = 100
    XList = []
    for x0 in X0:
        x = x0
        dg = fdm2c( cost, x )
        gnorm = np.linalg.norm( dg )
        xList = np.nan*np.ones( (n,kmax+1) )
        xList[:,0] = x[:,0]
        for k in range( kmax ):
            x = optvar.step( x, dg )
            dg = fdm2c( cost, x )
            gnorm = np.linalg.norm( fdm2c( cost, x ) )
            xList[:,k+1] = x[:,0]
            if gnorm < eps: break
        XList = XList + [xList[:,:k+1]]
        print( 'Complete for x0 (%s):' % k, x0.T, '\t->\t', x.T )

    # Create snapshot lists.
    X = np.hstack( [xList[:,:-1] for xList in XList] )
    Y = np.hstack( [xList[:,1:] for xList in XList] )

    # Solve for Koopman operator.
    kvar = KoopmanOperator( observe )
    print( kvar.edmd( X, Y ) )

    # Test Koopman operator on training set.
    PSIList = []
    for x0 in X0:
        psi = kvar.obsX.lift( x0 )
        psiList = np.empty( (observe()['Nk'], kmax+1) )
        psiList[:,0] = psi[:,0]
        for k in range( kmax ):
            psi = kvar.K@psi
            psiList[:,k+1] = psi[:,0]
            if np.linalg.norm( psi ) > 15:
                print('Operator dynamics diverged.')
                break
        PSIList = PSIList + [psiList]
        print( 'Complete for x0:', x0.T, '\t->\t', psi[:n].T )

    # Initialize plot variables.
    fig, axs = plt.subplots()

    # Add level set contour lines.
    eta = 20
    xBound = (-5, 5);  yBound = (-4, 4)
    xRange = np.linspace( xBound[0], xBound[1], 1000 )
    yRange = np.linspace( yBound[0], yBound[1], 1000 )
    xMesh, yMesh = np.meshgrid( xRange, yRange )
    gMesh = np.vstack( [
        cost( np.vstack( (xList, yList) ) )
            for xList, yList in zip( xMesh, yMesh ) ] )
    levels = [1] + [eta*(i + 1) for i in range( round( np.max( gMesh )/eta ) )]
    axs.contour( xMesh, yMesh, gMesh, levels=levels, colors='k' )

    # Add gradient descent results to plot.
    for xList in XList:
        axs.plot( xList[0,0], xList[1,0], marker='x', color='cornflowerblue' )
        axs.plot( xList[0], xList[1], color='cornflowerblue' )

    # Add operator results to plot.
    for psiList in PSIList:
        print( psiList )
        axs.plot( psiList[0,0], psiList[1,0], marker='x', color='indianred' )
        axs.plot( psiList[0], psiList[1], color='indianred' )

    # Display plot.
    axs.set_aspect('equal', adjustable='box')
    axs.grid( 1 )
    fig.tight_layout()
    plt.show( block=0 )
    input( 'Press ENTER to exit program... ' )
