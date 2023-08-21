import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman')
sys.path.insert(0, expanduser('~')+'/prog/geom')

import numpy as np

from KMAN.DataSets import *
from GEOM.Vehicle2D import *

# Hyper parameter(s)
dt = 0.01
Nx = 2
Nu = 2
Na = 4
# q = np.array( [ [10], [-2.5] ] )
q = np.random.rand( 2,1 )

# Anchor list
A = 10
aList = 2*A*np.random.rand( 2,Na ) - A
# aList = np.array( [
#     [1, 1, -1, -1],
#     [1, -1, 1, -1]
# ] )


# Model function.
def model(x, u):
    return x + dt*u


# Control methods.
def control(x):
    return q - x

def anchorControl(x):
    # Calculate squared terms.
    a2 = sum( alternate( aList**2 ) )
    d2 = alternate( anchorMeasure( x )**2 )
    qTa = alternate( q.T@aList )

    # Return control.
    return np.sum( (d2 - a2 + 2*qTa), axis=1 )[:,None]


# Signed columns helper.
def alternate(a):
    na = np.empty( a.shape )
    for i in range( 1,Na+1 ):
        sgn = 1*(i%2!=0) - 1*(i%2==0)
        na[:,i-1] = sgn*a[:,i-1]
    return na


# Anchor measurement function.
def anchorMeasure(x):
    d = np.empty( (1,Na) )
    for i, a in enumerate(aList.T):
        d[:,i] = (x - a[:,None]).T@(x - a[:,None])
    return np.sqrt( d )


# Main execution block.
if __name__ == '__main__':
    x0 = np.random.rand( 2,1 )

    print( 'uTa:\n', np.sum( alternate( 2*control( x0 ).T@aList ), axis=1 ) )
    print( 'c:\n', anchorControl( x0 ) )
