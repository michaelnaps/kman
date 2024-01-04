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

# Dimension of system and other parameter(s).
n = 2
m = 100
alpha = 1e-3
gamma = alpha
beta = 100

# Local maximum and offset.
A = 2
xmax = np.array( [[-0.270845],[-0.923039]] )

# Nonconvex objective function.
def cost(x):
    m = x.shape[1]
    g = (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    return g.reshape(1,m)

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

# Model function.
def model(x):
    xp = x - alpha*costgrad(x)
    return xp

# Observation function.
def observe(x=None):
    p = 5
    if x is None:
        return {'Nk': n+1}
    psi = np.vstack( [x] + [1] )
    return psi

# Main execution block.
if __name__ == '__main__':
    # Optimization variable.
    eps = 1e-3
    optvar = Optimizer( cost, eps=eps )
    optvar.setStepSize( alpha ).setMaxIter( np.inf )

    # Initial guess and system size.
    X0 = (2*A*np.random.rand( n,m ) - A) + xmax

    # Solve optimization problem and save steps.
    kmax = 10000
    Xlist = []
    for x0 in X0.T:
        x = x0[:,None]
        dg = fdm2c( cost, x )
        X = np.nan*np.ones( (n,kmax+1) )
        X[:,0] = x[:,0]
        for k in range( kmax ):
            x = optvar.step( x, dg )
            dg = fdm2c( cost, x )
            X[:,k+1] = x[:,0]
            if np.linalg.norm( costgrad( x ) ) < eps:
                break
        Xlist = Xlist + [X[:,:k+1]]
        # print( 'Complete for x0 (%s):' % k, x0.T, '\t->\t', x.T )

    # Group initial conditions by their fixed points.
    xmins = np.array( [
        [3,-2.805118,-3.779310, 3.584428],
        [2, 3.131312,-3.283186,-1.848126]
    ] )
    indlist = {i:[] for i in range( 4 )}
    for i, X in enumerate( Xlist ):
        for j, xmin in enumerate( xmins.T ):
            if np.linalg.norm( X[:,-1] - xmin ) < eps:
                indlist[j] = indlist[j] + [i]

    # Create snapshot lists.
    Xtrain = [np.hstack( Xlist[indlist[i]][:,:-1] ) for i in indlist]
    Ytrain = [fp[0][:,1:] for fp in fpsets]

    # Solve for Koopman operator.
    kvarlist = [KoopmanOperator( observe ).edmd( X, Y )
        for X, Y in zip( Xtrain, Ytrain )]
    for i, kvar in enumerate( kvarlist ):
        print( 'K%s:' % i, kvar )

    # Initialize plot variables.
    fig, axs = plt.subplots()

    # Add level set contour lines.
    eta = 20
    xBound = (-5, 5);  yBound = (-4, 4)
    xRange = np.linspace( xBound[0], xBound[1], 1000 )
    yRange = np.linspace( yBound[0], yBound[1], 1000 )
    xMesh, yMesh = np.meshgrid( xRange, yRange )
    gMesh = np.vstack( [
        cost( np.vstack( (xlist, ylist) ) )
            for xlist, ylist in zip( xMesh, yMesh ) ] )
    levels = [1, 5] + [eta*(i + 1) for i in range( round( np.max( gMesh )/eta ) )]
    axs.contour( xMesh, yMesh, gMesh, levels=levels, colors='k' )

    # Add gradient descent results to plot.
    for xlist in Xlist:
        axs.plot( xlist[0,0], xlist[1,0], marker='x', color='cornflowerblue' )
        axs.plot( xlist[0], xlist[1], color='cornflowerblue' )

    # # Add operator results to plot.
    # for psilist in PSIlist:
    #     axs.plot( psilist[0,0], psilist[1,0], marker='x', color='indianred' )
    #     axs.plot( psilist[0], psilist[1], color='indianred' )

    # Display plot.
    axs.set_aspect('equal', adjustable='box')
    axs.grid( 1 )
    fig.tight_layout()
    plt.show()
