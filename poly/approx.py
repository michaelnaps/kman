import sys
from os.path import expanduser
sys.path.insert(0, expanduser( '~' )+'/prog/kman')
sys.path.insert(0, expanduser( '~' )+'/prog/mpc')

import numpy as np
import matplotlib.pyplot as plt

from KMAN.Operators import *
from MPC.Optimizer import fdm2c

# Hyper paramter(s).
dt = 0.01
n = 1
N = 2*n
P = 2

# Objective function and model.
def polyn(X):
	# return X**4 - 3*X**3 + X**2 + X
	return X**2

def model(X):
	X1p = X[0] - dt*fdm2c( polyn, X[0] )
	Xp = np.vstack( [
		X1p,
		polyn( X1p )
	])
	return Xp

# Observation sets.
def obs1(X=None):
	if X is None:
		return {'Nk': n}
	return polyn( X )

def obs2(X=None):
	if X is None:
		return {'Nk': P - 1}
	psi = np.array( [X**p for p in range( 1,P )] )
	return psi

def obs3(X=None):
	if X is None:
		return {'Nk': n}
	return X

if __name__ == '__main__':
	# Initial conditions.
	A = 10;  N0 = 2
	X0 = 2*A*np.random.rand( n,N0 ) - A

	# Simulate and collect data.
	Nt = 1000
	Xlist = np.empty( (N0,N,Nt) )
	for i, x0 in enumerate( X0.T ):
		x = x0[:,None]
		X = np.empty( (N,Nt) )
		X[:,0] = x0
		for k in range( Nt-1 ):
			x = model( x )
			X[:,k] = x[:,0]
		Xlist[i,:,:] = X

	# Plot simulation results.
	fig, axs = plt.subplots()
	for X in Xlist:
		print( X )
		axs.plot( X[0], X[1] )
	plt.show()