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


# Anchor set.
A = 2
aList = 2*A*np.random.rand( 2,Na ) - A
# aList = np.array( [
#     [1, 1, -1, -1],
#     [1, -1, 1, -1]
# ] )

# Reflection set.
rxList = [[0],[1]]*aList
ryList = [[1],[0]]*aList


# Anchor measurement functions.
def anchorMeasure(x):
    d = np.empty( (1,Na) )
    for i, a in enumerate( aList.T ):
        d[:,i] = (x - a[:,None]).T@(x - a[:,None])

    print( d )

    return np.sqrt( d )

def reflectionMeasure(x, axis=0):
    if axis:
        rList = rxList
    else:
        rList = ryList

    dr = np.empty( (1,Na) )
    for i, r in enumerate( rList.T ):
        dr[:,i] = (x - r[:,None]).T@(x - r[:,None])

    print( dr )

    return np.sqrt( dr )


# Model function.
def model(x, u):
    return x + dt*u


# Control methods.
def control(x):
    return q - x

def anchorControl(x):
    # Calculate squared terms.
    arList = np.hstack( (aList, -rxList) )
    a2 = np.sum( arList**2, axis=0 )
    d2 = np.hstack( (anchorMeasure( x )**2, -reflectionMeasure( x )**2) )
    qTa = q.T@arList

    # Return control.
    return np.sum( (d2 - a2 + 2*qTa), axis=1 )[:,None]


# Signed columns helper.
def alternate(a):
    na = np.empty( a.shape )
    for i in range( 1,Na+1 ):
        sgn = 1*(i%2!=0) - 1*(i%2==0)
        na[:,i-1] = sgn*a[:,i-1]
    return na


# Main execution block.
if __name__ == '__main__':
    x0 = np.random.rand( 2,1 )

    arList = np.hstack( (aList, -rxList) )

    print( 'uTa:\n', 2*control( x0 ).T@np.sum( arList, axis=1 ) )
    print( 'c:\n', anchorControl( x0 ) )
