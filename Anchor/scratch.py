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
def getReflectionSet( axis=0 ):
    rList = np.zeros( (2,Na) )
    rList[1*(not axis),:] = aList[1*(not axis)]
    return rList

print( aList.T )
print( getReflectionSet().T )


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

def anchorControl(x, axis=0):
    # Get reflection set values.
    rList = getReflectionSet( axis=axis )

    # Calculate squared terms.
    dr2 = reflectionMeasure( x, rList )**2
    d2 = anchorMeasure( x )**2
    a2 = aList[axis,None]**2
    qa = q[axis,None]*aList[axis,None]
    asum = np.sum( aList[axis,None], axis=1 )

    # Return control.
    return np.sum( (d2 - dr2 - a2 + 2*qa)/(2*asum), axis=1 )


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

    print( 'ideal control: ', control( x0 ).T )
    print( 'anchor control:', np.vstack( (anchorControl( x0, axis=0 ), anchorControl( x0, axis=1 )) ).T )
