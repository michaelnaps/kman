import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman/Helpers')

import numpy as np
import matplotlib.pyplot as plt

from LearningStrategies import *
from Operators import *

# hyper paramters
dt = 0.001;
A = 10;
Nx = 2;

# model function: continuous, second-order
def model(x):
	a = 1.5;
	b = 2.0;
	dx = np.array( [
		a*(x[0]),
		b*(x[1] - x[0]**2)
	] );
	return dx;

# observation list
def obs(X=None):
	if X is None:
		meta = {'Nk': Nx+1};
		return meta;
	Psi = np.array( [
		X[0],
		X[1],
		X[0]**2
	] );
	return Psi;

if __name__=="__main__":
	# Generating random positions to learn from.
	N0 = 3;
	X = 2*A*np.random.rand( Nx,N0 ) - A;

	# Take derivative of positions using model.
	Y = np.empty( (Nx, N0) );
	for i, x in enumerate( X.T ):
		Y[:,i] = model( x[:,None] )[:,0];

	# Generate Lie operator from data.
	lvar = LieOperator( obs );
	lvar.edmd( X,Y );
	print( lvar );

	# Initial condition comparison.
	x0 = np.array( [[0.1], [0.1]] );
	Psi0 = obs( x0 );

	# Simulation time parameters.
	T = 1;  dt = 0.001;
	Nt = round(T/dt) + 1;
	tList = [ [i*dt for i in range( Nt )] ];

	# Discrete model functions.
	mDiscrete = lambda x: x + dt*model( x );
	lDiscrete = lambda Psi: Psi + dt*(lvar.L@Psi);

	# Initialize matrices and set initial point.
	xList = np.empty( (Nx,Nt) );
	psiList = np.empty( (lvar.obsY.Nk,Nt) );
	xList[:,0] = x0[:,0];
	psiList[:,0] = Psi0[:,0];

	# Simulate using discrete functions.
	for i in range( Nt-1 ):
		xList[:,i+1] = mDiscrete( xList[:,i,None] )[:,0];
		psiList[:,i+1] = lDiscrete( psiList[:,i,None] )[:,0];

	# Plot results of the simulation.
	fig, axs = plt.subplots( 3,1 );
	axs[0].plot(tList[0], xList[0], label='Model');
	axs[0].plot(tList[0], psiList[0], linestyle='--', label='Lie');
	axs[0].set_title( '$x_1$' );

	axs[1].plot(tList[0], xList[1], label='Model');
	axs[1].plot(tList[0], psiList[1], linestyle='--', label='Lie');
	axs[1].set_title( '$x_2$' );

	axs[2].plot(tList[0], xList[0]**2, label='Model');
	axs[2].plot(tList[0], psiList[2], linestyle='--', label='Lie');
	axs[2].set_title( '$x_1^2$' );

	axs[0].legend();
	fig.tight_layout();
	plt.show();