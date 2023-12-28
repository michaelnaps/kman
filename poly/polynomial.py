# ============================================================================ #
# Polynomial Functions in Terms of Koopman/Lie Operators
# Dynamical system and observable selection techniques taken from:
#	Brunton SL, Brunton BW, Proctor JL, Kutz JN (2016) Koopman Invariant
#	Subspaces and Finite Linear Representations of Nonlinear Dynamical Systems
#	for Control. PLoS ONE 11(2): e0150171. doi:10.1371/journal.pone.0150171
# ============================================================================ #

import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman')

import numpy as np
import matplotlib.pyplot as plt

from KMAN.Operators import *

# hyper paramters
dt = 0.001
A = 10
Nx = 2
W = 2

# model function: continuous, second-order
def model(x):
	a = 1.25
	b = 1.50
	dx = np.array( [
		a*(x[0]),
		b*(x[1] - x[0]**W)
	] )
	return dx

# observation list
def obsX(x=None):
	if x is None:
		meta = {'Nk': Nx+1}
		return meta

	Psi = np.array( [
		x[0],
		x[1],
		x[0]**2
	] )

	return Psi

if __name__=="__main__":
	# Generating random positions to learn from.
	N0 = 3
	X = 2*A*np.random.rand( Nx,N0 ) - A
	Y = X + dt*model( X )

	# Initialize L and K operator variables.
	kvar = KoopmanOperator( obsX )
	lvar = LieOperator( obsX )

	kvar.edmd( X, Y )
	lvar.K2L( kvar )
	lvar.resError( X, Y, save=1 )
	print( 'K:\n', kvar )
	print( 'L:\n', lvar )

	# Initial condition comparison.
	x0 = np.array( [[0.1], [0.1]] )
	Psi0 = obsX( x0 )

	# Simulation time parameters.
	T = 1;  Nt = round(T/dt) + 1
	tList = [ [i*dt for i in range( Nt )] ]

	# Discrete model functions.
	mDiscrete = lambda x: x + dt*model( x )
	lDiscrete = lambda Psi: Psi + dt*(lvar.L@Psi)

	# Initialize matrices and set initial point.
	xList = np.empty( (Nx,Nt) )
	LpsiList = np.empty( (lvar.obsY.Nk,Nt) )
	KpsiList = np.empty( (kvar.obsY.Nk,Nt) )

	xList[:,0] = x0[:,0]
	LpsiList[:,0] = Psi0[:,0]
	KpsiList[:,0] = Psi0[:,0]

	# Simulate using discrete functions.
	for i in range( Nt-1 ):
		xList[:,i+1] = mDiscrete( xList[:,i,None] )[:,0]
		LpsiList[:,i+1] = lDiscrete( LpsiList[:,i,None] )[:,0]
		KpsiList[:,i+1] = (kvar.K@KpsiList[:,i,None])[:,0]

	# Plot results of the simulation.
	fig, axs = plt.subplots( 3,1 )
	axs[0].plot(tList[0], xList[0], linewidth=3.0, label='Model')
	axs[0].plot(tList[0], KpsiList[0], linewidth=2.0, linestyle='--', label='Koopman')
	axs[0].plot(tList[0], LpsiList[0], linewidth=1.5, linestyle='--', label='Lie')
	axs[0].set_title( '$x_1$' )

	axs[1].plot(tList[0], xList[1], linewidth=3.0, label='Model')
	axs[1].plot(tList[0], KpsiList[1], linewidth=2.0, linestyle='--', label='Koopman')
	axs[1].plot(tList[0], LpsiList[1], linewidth=1.5, linestyle='--', label='Lie')
	axs[1].set_title( '$x_2$' )

	axs[2].plot(tList[0], xList[0]**2, linewidth=3.0, label='Model')
	axs[2].plot(tList[0], KpsiList[2], linewidth=2.0, linestyle='--', label='Koopman')
	axs[2].plot(tList[0], LpsiList[2], linewidth=1.5, linestyle='--', label='Lie')
	axs[2].set_title( '$x_1^2$' )

	axs[0].legend()
	fig.tight_layout()
	plt.show()
