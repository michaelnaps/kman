import sys
from os.path import expanduser
sys.path.insert(0, expanduser( '~' )+'/prog/kman')
sys.path.insert(0, expanduser( '~' )+'/prog/mpc')

import numpy as np
import matplotlib.pyplot as plt

from KMAN.Operators import *
from MPC.Optimizer import fdm2c

# Hyper paramter(s).
dt = 0.1
n = 1
N = 2
P = 2

# Objective function and model.
def polyn(X):
	# return X**4 - 3*X**3 + X**2 + X
	return X**2

def model(X):
	# Shape data properly.
	m = X.shape[1]
	X1 = X[0].reshape( n,m )

	# Propagate position.
	X1p = X1 - dt*fdm2c( polyn, X1 )
	Xp = np.vstack( [
		X1p,
		polyn( X1p )
	])
	return Xp

# Observation sets.
def obs3(X=None):
	if X is None:
		return {'Nk': n}
	m = X.shape[1]
	return X[0].reshape( n,m )

def obs23(X=None):
	if X is None:
		return {'Nk': obs3()['Nk'] + (P-1)}
	psi2 = np.array( [X[0]**p for p in range( 1,P )] )
	psi3 = obs3( X )
	return np.vstack( (psi2, psi3) )

def obs2(X=None):
	if X is None:
		return {'Nk': (P-1)}
	psi2 = np.array( [X[0]**p for p in range( 1,P )] )
	return psi2

def obs123(X=None):
	if X is None:
		return {'Nk': obs23()['Nk'] + n}
	psi1 = X[1]
	psi23 = obs23( X )
	return np.vstack( (psi1, psi23) )

# Shift functions.
def shift23(Klist):
	p3 = obs3()['Nk']
	p23 = obs23()['Nk']
	T = np.eye( p23, p23 )
	T[p23-p3:,p23-p3:] = Klist[0].K
	return T

def shift12(Klist):
	p3 = obs3()['Nk']
	p23 = obs23()['Nk']
	p123 = obs123()['Nk']
	T = np.eye( p123, p123 )
	T[p123-p23:p123-p3,p123-p23:] = Klist[0].K
	T[p123-p3:,p123-p3:] = Klist[1].K
	return T

if __name__ == '__main__':
	# Initial conditions.
	A = 10;  N0 = 25
	X0 = 2*A*np.random.rand( n,N0 ) - A
	X0 = np.vstack( (X0, polyn( X0 )) )

	# Simulate and collect data.
	Nt = 1000
	Xlist = np.empty( (N0,N,Nt) )
	for i, x0 in enumerate( X0.T ):
		x = x0[:,None]
		X = np.empty( (N,Nt) )
		X[:,0] = x0
		for k in range( Nt-1 ):
			x = model( x )
			X[:,k+1] = x[:,0]
		Xlist[i,:,:] = X

	# Create cascade variables.
	kvar3 = KoopmanOperator( obs3 )
	kvar2 = KoopmanOperator( obs23, obs2, T=shift23( (kvar3,) ) )
	kvar1 = KoopmanOperator( obs123, T=shift12( (kvar2, kvar3) ) )

	# Form data sets for cascade.
	X = np.hstack( [X[:,:-1] for X in Xlist] )
	Y = np.hstack( [X[:,1:] for X in Xlist] )
	Xsets = (X, X, X)
	Ysets = (Y, Y, Y)

	# Perform cascade EDMD.
	Klist = (kvar1, kvar2, kvar3)
	Tlist = (shift12, shift23)
	Klist = cascade_edmd( Tlist, Klist, Xsets, Ysets, X0 )
	for i, kvar in enumerate( Klist ):
		print( 'K%s:'%i, kvar )

	# Plot simulation results.
	fig, axs = plt.subplots()
	for X in Xlist:
		axs.plot( X[0], X[1] )
	plt.show()
