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

# Convex objective function.
def cost(x):
    g = x.T@x
    return g

# Main execution block.
if __name__ == '__main__':
    # Optimization variable.
    n = 100
    optvar = Optimizer( cost, eps=1e-21 ).setMaxIter( np.inf )

    # Initial guess and system size.
    A = 10
    x0 = 2*A*np.random.rand( n,1 ) - A
    print( 'x0:\n', x0 )

    # Solve optimization problem (convex).
    print( 'x*:', optvar.solve( x0, verbose=0 ) )
