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

# Anchor control constants.
D = 1/2*np.diag( np.sum( aList, axis=1 ) )
Q = D@np.sum( 2*q*aList + aList**2, axis=1 )[:,None]

print( 'D', D )

# Reflection set.
def getReflectionSet( axis=0 ):
    rList = np.zeros( (2,Na) )
    rList[1*(not axis),:] = aList[1*(not axis)]
    return rList


# Anchor measurement functions.
def anchorMeasure(x):
    d = np.empty( (1,Na) )
    for i, a in enumerate( aList.T ):
        d[:,i] = (x - a[:,None]).T@(x - a[:,None])
    return np.sqrt( d )

def reflectionMeasure(x, rList):
    dr = np.empty( (1,Na) )
    for i, r in enumerate( rList.T ):
        dr[:,i] = (x - r[:,None]).T@(x - r[:,None])
    return np.sqrt( dr )


# Model function.
def model(x, u):
    return x + dt*u


# Control methods.
def control(x):
    return q - x

def anchorControl(x):
    # Reflection set.
    rxList = getReflectionSet( 0 )
    ryList = getReflectionSet( 1 )

    # Calculate squared terms.
    d2 = anchorMeasure( x )**2
    d = np.vstack( (
        np.sum( d2 + reflectionMeasure( x, rxList )**2 ),
        np.sum( d2 + reflectionMeasure( x, ryList )**2 )
    ) )

    print( 'd', d )
    print( 'Dd', D@d )
    print( 'Q', Q )

    # Return control.
    return D@d + Q


# Main execution block.
if __name__ == '__main__':
    x0 = np.random.rand( 2,1 )

    print( 'ideal control: ', control( x0 ).T )
    print( 'anchor control:', anchorControl( x0 ).T )
