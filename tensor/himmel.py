# Path imports.
import sys
from os.path import expanduser
sys.path.insert( 0, expanduser('~')+'/prog/kman' )  # Koopman operator classes.
sys.path.insert( 0, expanduser('~')+'/prog/four' )  # Fourier transform classes.
sys.path.insert( 0, expanduser('~')+'/prog/mpc' )   # Optimization classes.

# Standard imports.
import numpy as np
import matplotlib.pyplot as plt

# Homemade imports.
from KMAN.Operators import *
from FOUR.Transforms import *
from MPC.Optimizer import *

# Set global number print setting.
np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)

# Dimension of system and other parameter(s).
n = 2
m = 150
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

def costnorm(x):
    dgnorm = np.linalg.norm( costgrad( x ) )
    return dgnorm

# Model function.
def model(x):
    xp = x - alpha*costgrad(x)
    return xp

# Observation function.
def obs(x=None):
    if x is None:
        return {'Nk': n+1}
    m = x.shape[1]
    psi = np.vstack( (x, np.ones( (1,m) )) )
    return psi

# Shape functions.
def koopmanStack(Klist):
    # Stack operators.
    l = len( Klist )
    p, q = Klist[0].shape

    # Main execution loop.
    Kstack = np.empty( (p*q, l) )
    for k, K in enumerate( Klist ):
        Kstack[:,k] = K.reshape( p*q, )

    # Return grid stack.
    return Kstack

def koopmanSort(X):
    M = X.shape[1]
    indexlist = [None for i in range( M )]
    for i, x in enumerate( X.T ):
        if x[0] > xmax[0] and x[1] > xmax[1]:
            indexlist[i] = 0
        elif x[0] < xmax[0] and x[1] > xmax[1]:
            indexlist[i] = 1
        elif x[0] < xmax[0] and x[1] < xmax[1]:
            indexlist[i] = 2
        elif x[0] > xmax[0] and x[1] < xmax[1]:
            indexlist[i] = 3
    return indexlist

# Main execution block.
if __name__ == '__main__':
    # Optimization variable.
    eps = 1e-3
    optvar = Optimizer( cost, eps=eps )
    optvar.setStepSize( alpha ).setMaxIter( np.inf )

    # Initial guess and system size.
    X0 = (2*A*np.random.rand( n,m ) - A) + xmax

    # Identifying minima empirically.
    Xminima = np.array( [
        [3,-2.805118,-3.779310, 3.584428],
        [2, 3.131312,-3.283186,-1.848126]
    ] )
    Xindex = [[] for _ in range( 4 )]

    # Generate data using steepest-descent method.
    Xlist = []
    for x0 in X0.T:
        x = x0[:,None]
        X = [x]
        while costnorm( x ) > eps:
            x = model( x )
            X = X + [x]
        Xlist = Xlist + [np.hstack( X )]

    # Separate initial positions based on final positions.
    for i, X in enumerate( Xlist ):
        for j, xmin in enumerate( Xminima.T ):
            if np.linalg.norm( X[:,-1] - xmin ) < eps:
                Xindex[j] = Xindex[j] + [i]
    assert np.hstack( Xindex ).shape[0] == m, \
        f"({m - np.hstack( Xindex ).shape[0]}) initial position(s) not set."

    # Generate training sets for Koopman sub-domains.
    Xtrain = [np.hstack( [Xlist[i][:,:-1] for i in indlist] )
        for indlist in Xindex]
    Ytrain = [np.hstack( [Xlist[i][:,1:] for i in indlist] )
        for indlist in Xindex]

    # Solve for Koopman operator.
    kvarlist = [KoopmanOperator( obs ).edmd( X, Y )
        for X, Y in zip( Xtrain, Ytrain )]
    for i, kvar in enumerate( kvarlist ):
        print( 'K%s:' % (i + 1), kvar )
        print( 'Eig:', np.linalg.eig( kvar.K )[0] )
        print( '---' )

    # Create Fourier transform mesh on x,y-axes.
    L = 100  # Number of grid points.
    xbound = (-5, 5);  ybound = (-4, 4)
    xrange = np.linspace( xbound[0], xbound[1], L )
    yrange = np.linspace( ybound[0], ybound[1], L )
    Xmesh = np.hstack( [
        np.hstack( [np.vstack( (x, y) ) for x in xrange] )
            for y in yrange] )
    Kindex = koopmanSort( Xmesh )
    Ksort = [kvarlist[i].K for i in Kindex]
    Kmesh = koopmanStack( Ksort )

    # Solve Fourier transform.
    Fvar = RealFourier( Xmesh, Kmesh ).dmd( N=250 )
    print( "Computed transform." )

    def koopmanSolve(X):
        Nx = X.shape[1]
        Nk = obs()['Nk']
        Klist = Fvar.solve( X ).reshape( Nk,Nk ) if Nx == 1 \
            else Fvar.solve( X ).T.reshape( Nx,Nk,Nk )
        return Klist

    # # Simulate with Koopman tensor results.
    # M = 1000
    # P0 = obs( X0 )
    # Plist = []
    # for p0 in P0.T:
    #     p = p0[:,None]
    #     P = [p]
    #     for _ in range( M ):
    #         K = koopmanSolve( p[:n] )
    #         if np.linalg.norm( p - K@p ) < 1e-6:
    #             break
    #         p = K@p
    #         # if np.linalg.norm( p ) > 100:
    #         #     P = [p0[:,None]]
    #         #     break
    #         P = P + [p]
    #     Plist = Plist + [np.hstack( P )]

    # Initialize plot variables.
    fig, axs = plt.subplots()

    # Add level set contour lines.
    eta = 20
    xmesh, ymesh = np.meshgrid( xrange, yrange )
    gmesh = np.vstack( [
        cost( np.vstack( (xlist, ylist) ) )
            for xlist, ylist in zip( xmesh, ymesh ) ] )
    levels = [1, 5] + [eta*(i + 1) for i in range( round( np.max( gmesh )/eta ) )]
    axs.contour( xmesh, ymesh, gmesh, levels=levels, colors='k' )

    # Koopman coefficient mesh.
    indexlist = [(0,0)]
    kmesh = [np.vstack( [
        [koopmanSolve( np.vstack( (x, y) ) )[i]
            for x, y in zip( xlist, ylist )]
                for xlist, ylist in zip( xmesh, ymesh ) ] )
                    for i in indexlist]
    for klist in kmesh:
        axs.contour( xmesh, ymesh, klist, colors='indianred' )

    # Add gradient descent results to plot.
    for X in Xlist:
        axs.plot( X[0,0], X[1,0], marker='x', color='cornflowerblue' )
        axs.plot( X[0], X[1], color='cornflowerblue' )

    # # Add Koopman operator results to plot.
    # for P in Plist:
    #     axs.plot( P[0,0], P[1,0], marker='x', color='indianred' )
    #     axs.plot( P[0], P[1], color='indianred' )

    # Display plot.
    axs.set_aspect('equal', adjustable='box')
    axs.grid( 1 )
    fig.tight_layout()
    plt.show()
