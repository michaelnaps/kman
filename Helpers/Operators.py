import numpy as np
from LearningStrategies import *

def cascade_edmd(Klist, Tlist, X, Y, X0):
	pass;

class Observables:
	def __init__(self, obs):
		self.obs = obs;
		self.Nk = obs['Nk'];

	def lift(self, x):
		return self.obs(x);

	def liftData(self, X, X0, obs=None):
		if obs is None:
			obs = self.obs;

		# Dimension variable definitions.
		N = len( X0 );
		K = len( X[0] );
		M = len( X0[0] );

		# Lifted state initialization
		Psi = np.empty( (Nk, K*M) );

		for n in range( K*M ):
			Psi[:,n] = obs( X[:,n,None] )[:,0];

		return Psi;

class KoopmanOperator(LearningStrategies):
	def __init__(self, obsX, obsY=None, T=None, K=None):
		pass;