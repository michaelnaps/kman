import numpy as np
import matplotlib.pyplot as plt

from Helpers.LearningStrategies import *
from Helpers.Operators import *

# hyper paramters
dt = 0.001;
A = 10;
Nx = 2;

# model function: continuous, second-order
def model(x):
	a = 1.00;
	b = 0.25;
	dx = np.array( [
		a*(x[0]),
		b*(x[1] - x[0]**2)
	] );
	return dx;

# observation list
def obsX(X=None):
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
	X = np.random.rand( Nx,N0 );

	# Take derivative of positions using model.
	Y = np.empty( (Nx, N0) );
	for i, x in enumerate( X.T ):
		Y[:,i] = model( x[:,None] )[:,0];

	# Generate Lie operator from data.
	lvar = LieOperator( obs );
	lvar.edmd(X, Y);
	print( lvar );

	# Initial condition comparison.
	x0 = np.array( [[0.1], [0.999]] );
	Psi0 = obs( x0 );

	# Simulation time parameters.
	T = 10;  dt = 0.001;
	Nt = round(T/dt) + 1;
	tList = [ [i*dt for i in range( Nt )] ];

	# Discrete model functions.
	mDiscrete = lambda x: x + dt*model( x );
	kDiscrete = lambda Psi: Psi + dt*(lvar.L@Psi);

	# Initialize matrices and set initial point.
	xList = np.empty( (Nx,Nt) );
	psiList = np.empty( (lvar.obsX.Nk,Nt) );
	xList[:,0] = x0[:,0];
	psiList[:,0] = Psi0[:,0];

	# Simulate using discrete functions.
	for i in range( Nt-1 ):
		if i % 100 == 0:
			print( xList[:,i] );
			print( psiList[:,i] );
		xList[:,i+1] = mDiscrete( xList[:,i,None] )[:,0];
		psiList[:,i+1] = kDiscrete( psiList[:,i,None] )[:,0];

	# Plot results of the simulation.
	fig, axs = plt.subplots();
	axs.plot(xList[0,:], xList[1,:], label='Model');
	axs.plot(psiList[0,:], psiList[1,:], label='Lie');
	plt.legend();
	plt.show();