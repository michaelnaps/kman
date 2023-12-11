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

# Dimension of system.
n = 2
beta = 100

# Convex objective function.
def cost(x):
    g = (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    return g

# Observation function.
def observe(x=None):
    if x is None:
        return {'Nk': n}
    psi = x
    return psi

# Main execution block.
if __name__ == '__main__':
    # Optimization variable.
    eps = 1e-21
    alpha = 1e-3
    optvar = Optimizer( cost, eps=eps )
    optvar.setStepSize( alpha ).setMaxIter( np.inf )

    # Initial guess and system size.
    p = 1
    A = 1
    X0 = np.array( [ [[3],[2]] ] ) \
        + 2*A*np.random.rand( n,p ) - A

    # Solve optimization problem and save steps.
    XList = []
    qmax = 1000
    for x0 in X0:
        x = x0
        xList = [x]
        dg = fdm2c( cost, x )
        gnorm = np.linalg.norm( dg )
        q = 0
        while gnorm > 1e-21 and q < qmax:
            x = optvar.step( x, dg )
            xList = xList + [x]
            dg = fdm2c( cost, x )
            gnorm = np.linalg.norm( fdm2c( cost, x ) )
            q += 1
        XList = XList + [np.hstack( xList )]
        print( 'Complete for x0:', x0.T )
    print( 'xList:', [xList[:,-1] for xList in XList] )

    # Create snapshot lists.
    X = np.hstack( [xList[:,:-1] for xList in XList] )
    Y = np.hstack( [xList[:,1:] for xList in XList] )

    # Solve for Koopman operator.
    kvar = KoopmanOperator( observe )
    kvar.edmd( X, Y )
    print( kvar )
