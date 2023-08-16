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
Nf = 20  # Fourier expansion number.


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
        return {'Nk': 2}
    psi3 = np.vstack( (np.sin( X[2] ), np.cos( X[2] )) )
    return psi3

def obs3(X=None):
    if X is None:
        return {'Nk': 2*Nf+1}
    xSin = [ np.sin( k*X[0] ) for k in range( 1,Nf+1 ) ]
    xCos = [ np.cos( k*X[0] ) for k in range( Nf+1 ) ]
    psi3 = np.vstack( (xSin, xCos) )
    return psi3

def obs3p(X=None):
    if X is None:
        return {'Nk': 1}
    psi3p = X[0,None]**3
    return psi3p

def obs123(X=None):
    if X is None:
        return {'Nk': obs1()['Nk']+obs2()['Nk']+obs3()['Nk']}
    psi1 = obs1( X )
    psi2 = obs2( X )
    psi3 = obs3( X )
    psi = np.vstack( (psi1, psi2, psi3) )
    return psi

def obs123p(X=None):
    if X is None:
        return {'Nk': obs1()['Nk']+obs2()['Nk']+obs3p()['Nk']}
    psi1 = obs1( X )
    psi2 = obs2( X )
    psi3p = obs3p( X )
    psip = np.vstack( (psi1, psi2, psi3p) )
    return psip


# Main execution block.
if __name__ == '__main__':
    # Initialize time-series data.
    T = 1;  Ntt = round( T/dt ) + 1
    tTrain = np.array( [ [i*dt for i in range( Ntt )] ] )

    # State initialization for training.
    A = 3.0
    N0t = 10
    X0t = np.vstack( (
        2*A*np.random.rand( Nx-1, N0t ) - A,  # position init
        np.zeros( (Nx-2, N0t ) )              # time-series init
    ) )

    # Generate state data sets.
    xData = generate_data( tTrain, model, X0t )[0]
    xTrain = stack_data( xData[:,:-1], N0t, Nx, Ntt-1 )
    yTrain = stack_data( xData[:,1:], N0t, Nx, Ntt-1 )

    # Cascade sets.
    X = (xTrain, xTrain, xTrain)
    Y = (yTrain, yTrain, xTrain)

    # Initialize shift functions.
    p1 = obs1()['Nk'];   q1 = obs1()['Nk']
    p2 = obs2()['Nk'];   q2 = obs2()['Nk']
    p3 = obs3p()['Nk'];  q3 = obs3()['Nk']
    def shift( Klist ):
        T = np.eye( p1+p2+p3, q1+q2+q3 )
        T[p1:p1+p2,q1:q1+q2] = Klist[0].K
        T[p1+p2:,q1+q2:] = Klist[1].K
        return T

    # Initialize operator variables and solve.
    k3var = KoopmanOperator( obs3, obs3p )
    k2var = KoopmanOperator( obs2 )
    k1var = KoopmanOperator( obs123, obs123p, T=shift( (k2var, k3var) ) )

    Klist = (k1var, k2var, k3var)
    Tlist = (shift, None)
    Klist = cascade_edmd(Tlist, Klist, X, Y, X0t)
    print('Cascade EDMD Complete.')

    for K in Klist:
        print( K )

    # Calculate cumulative operator.
    kvar = KoopmanOperator( obs123, obs123p )
    kvar.setOperator( Klist[0].K@shift( Klist[1:] ) )
    kvar.resError( xTrain, yTrain, save=1 )
    print( 'Cumulative operator:\n', kvar )
