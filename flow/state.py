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

# Convex objective function.
def cost(x):
    g = x.T@x
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
    optvar = Optimizer( cost, eps=eps ).setMaxIter( np.inf )

    # Initial guess and system size.
    p = 2
    A = 10
    X0 = 2*A*np.random.rand( p,n,1 ) - A

    # Solve optimization problem and save steps.
    XList = []
    qmax = np.inf
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

    # Create snapshot lists.
    X = np.hstack( [xList[:,:-1] for xList in XList] )
    Y = np.hstack( [xList[:,1:] for xList in XList] )

    # Solve for Koopman operator.
    kvar = KoopmanOperator( observe )
    kvar.edmd( X, Y )
    print( kvar )
