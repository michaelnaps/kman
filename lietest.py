import numpy as np

from Helpers.LearningStrategies import *
from Helpers.Operators import *

# hyper paramters
dt = 0.001;
A = 10;
Nx = 2;

# model function: continuous, second-order
def model(x):
	a = 3;
	b = 4;
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
	X = np.random.rand( Nx,N0 );

	# Take derivative of positions using model.
	Y = np.empty( (Nx, N0) );
	for i, x in enumerate( X.T ):
		Y[:,i] = model( x[:,None] )[:,0];

	# Generate Lie operator from data.
	lvar = LieOperator( obs );
	lvar.edmd(X, Y);
	print( lvar );
