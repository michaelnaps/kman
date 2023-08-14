# Path imports.
import sys
from os.path import expanduser
sys.path.insert( 0, expanduser('~')+'/prog/kman' )
sys.path.insert( 0, expanduser('~')+'/prog/geom' )

# Standard imports.
import numpy as np
from KMAN.Operators import *
from GEOM.Vehicle2D import *


# Hyper parameters.
dt = 0.001
Nx = 3
Nf = 10


# Duffing model function.
def model(x):
    c = [ 1, 1, 0.07, 0.20, 1.10 ]
    dx = np.array( [
        x[1],
        c[0]*x[0] - c[1]*x[0]**3 - c[2]*x[1] - c[3]*np.cos( c[4]*x[2] ),
        1
    ] )
    return x + dt*dx


# Observation functions.
def obs1(X=None):
    if X is None:
        return {'Nk': Nx-1}
    psi = X[:Nx-1]
    return psi

def obs2(X=None):
    if X is None:
        return {'Nk': 2*Nf}
    xSin = [ np.sin( 2*np.pi*k*X[0]/dt ) for k in range( Nf ) ]
    xCos = [ np.cos( 2*np.pi*k*X[0]/dt ) for k in range( Nf ) ]
    psi = np.vstack( (xSin, xCos) )
    return psi

def obs3(X=None):
    if X is None:
        return {'Nk': 2}
    psi = np.vstack( (np.sin( X[2] ), np.cos( X[2] )) )
    return psi