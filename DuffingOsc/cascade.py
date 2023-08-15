# Path imports.
import sys
from os.path import expanduser
sys.path.insert( 0, expanduser('~')+'/prog/kman' )  # Koopman operator classes.
sys.path.insert( 0, expanduser('~')+'/prog/four' )  # Fourier series classes.
sys.path.insert( 0, expanduser('~')+'/prog/geom' )  # Plotting and sim classes.

# Standard imports.
import numpy as np
import matplotlib.pyplot as plt
from duffing import *
from KMAN.Operators import *
from FOUR.Transforms import *
from GEOM.Vehicle2D import *


# Hyper parameters.
dt = 0.001
Nx = 3
Nf = 10  # Fourier expansion number.


# Model function.
model = lambda x: model3(x, dt=dt)


# Observation functions.
def obs1(X=None):
    if X is None:
        return {'Nk': Nx-1}
    psi1 = X[:Nx-1]
    return psi1

def obs2(X=None):
    if X is None:
        return {'Nk': 2*(Nf+1)}
    xSin = [ np.sin( k*X[0] ) for k in range( Nf+1 ) ]
    xCos = [ np.cos( k*X[0] ) for k in range( Nf+1 ) ]
    psi2 = np.vstack( (xSin, xCos) )
    return psi2

def obs2p(X=None):
    if X is None:
        return {'Nk': 1}
    psi2p = X[0,None]**3
    return psi2p

def obs3(X=None):
    if X is None:
        return {'Nk': 2}
    psi3 = np.vstack( (np.sin( X[2] ), np.cos( X[2] )) )
    return psi3

def obs23(X=None):
    if X is None:
        return {'Nk': obs2()['Nk']+obs3()['Nk']}
    psi2 = obs2( X )
    psi3 = obs3( X )
    psi23 = np.vstack( (psi2, psi3) )
    return psi23

def obs23p(X=None):
    if X is None:
        return {'Nk': obs2p()['Nk']+obs3()['Nk']}
    psi2p = obs2p( X )
    psi3 = obs3( X )
    psi23p = np.vstack( (psi2p, psi3) )
    return psi23p

def obs(X=None):
    if X is None:
        return {'Nk': obs1()['Nk']+obs2()['Nk']+obs3()['Nk']}
    psi1 = obs1( X )
    psi2 = obs2( X )
    psi3 = obs3( X )
    psi = np.vstack( (psi1, psi2, psi3) )
    return psi

def obsp(X=None):
    if X is None:
        return {'Nk': obs1()['Nk']+obs2p()['Nk']+obs3()['Nk']}
    psi1 = obs1( X )
    psi2p = obs2p( X )
    psi3 = obs3( X )
    psip = np.vstack( (psi1, psi2p, psi3) )
    return psip


# Main execution block.
if __name__ == '__main__':
    # Initialize time-series data.
    T = 10;  Ntt = round( T/dt ) + 1
    tTrain = np.array( [ [i*dt for i in range( Ntt )] ] )

    # State initialization for training.
    A = 1.0
    N0t = 1
    X0t = np.vstack( (
        2*A*np.random.rand( Nx-1, N0t ) - A,  # position init
        np.zeros( (Nx-2, N0t ) )              # time-series init
    ) )

    # Generate state data sets.
    xData = generate_data( tTrain, model, X0t )[0]
    xTrain = stack_data( xData[:,:-1], N0t, Nx, Ntt-1 )
    yTrain = stack_data( xData[:,1:], N0t, Nx, Ntt-1 )

    # Initialize shift functions.
    p1 = obs1()['Nk'];  q1 = obs1()['Nk']
    p2 = obs2p()['Nk'];  q2 = obs2()['Nk']
    p3 = obs3()['Nk'];  q3 = obs3()['Nk']
    def shift3(k2var):
        T2 = np.eye( p1+p2+p3, q1+q2+q3 )
        T2[p1:p1+p2,q1:q1+q2] = k2var.K
        return T2
    def shift2(k3var):
        T3 = np.eye( p1+p2+p3, q1+q2+q3 )
        T3[p1+p2:,q1+q2:] = k3var.K
        return T3

    # Initialize operator variables and solve.
    k3var = KoopmanOperator( obs2, obs2p )
    k2var = KoopmanOperator( obs23, obs23p, T=shift3(k3var) )
    print( k2var )
    k1var = KoopmanOperator( obs, obsp, T=shift2(k2var) )

    Klist = (k1var, k2var, k3var)
    Tlist = (shift3, shift2)
    Klist = cascade_edmd(Klist, Tlist, xTrain, yTrain, X0)
    print('Cascade EDMD Complete.')

    # # Initialize and solve for Koopman operators.
    # k2var = KoopmanOperator( obs2, obs2p )
    # print( 'K: x0(10) in [%.1f,' % -A, '%.1f]' % A )
    # print( k2var.edmd( xTrain, xTrain ) )
    # print( shift3( k2var ).shape )

    # k3var = KoopmanOperator( obs3 )
    # print( 'K: x0(10) in [%.1f,' % -A, '%.1f]' % A )
    # print( k3var.edmd( xTrain, yTrain ) )
    # print( shift2( k3var ).shape )
