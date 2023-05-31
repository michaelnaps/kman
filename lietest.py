import numpy as np

from Helpers.LearningStrategies import *
from Helpers.Operators import *

# hyper paramters
A = 10;
Nx = 2;

# model function: continuous, second-order
def model(x):
	dx = np.array( [
		x[0],
		x[1] - x[0]**2
	] );
	return dx;

if __name__=="__main__":
	N0 = 2;
	X0 = 2*A*np.random.rand(Nx, N0) - A;
