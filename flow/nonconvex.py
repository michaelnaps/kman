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
    return g

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
    p = 3
    if x is None:
        return {'Nk': (p + 1)*n}
    psi = np.vstack( [x] + [gamma**(k+1)*costgradprop( x, p=k ) for k in range( p )] )
    return psi

# Main execution block.
if __name__ == '__main__':
    # Optimization variable.
    eps = 1e-21
    optvar = Optimizer( cost, eps=eps )
    optvar.setStepSize( alpha ).setMaxIter( np.inf )

    # Initial guess and system size.
    p = 10
    A = 1
    X0 = 2*A*np.random.rand( p,n,1 ) - A

    # Solve optimization problem and save steps.
    XList = []
    kmax = 2500
    for x0 in X0:
        x = x0
        xList = [x]
        dg = fdm2c( cost, x )
        gnorm = np.linalg.norm( dg )
        k = 0
        while gnorm > 1e-21 and k < kmax:
            x = optvar.step( x, dg )
            xList = xList + [x]
            dg = fdm2c( cost, x )
            gnorm = np.linalg.norm( fdm2c( cost, x ) )
            k += 1
        XList = XList + [np.hstack( xList )]
        print( 'Complete for x0:', x0.T, '\t->\t', x.T )

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
        psiList = [psi]
        for _ in range( kmax if np.isfinite( kmax ) else 10000 ):
            psi = kvar.K@psi
            psiList = psiList + [psi]
        PSIList = PSIList + [np.hstack( psiList )]
        print( 'Complete for x0:', x0.T, '\t->\t', psi[:n].T )
