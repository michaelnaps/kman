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
		# If obs not given, use class.
		if obs is None:
			obs = self.obs;

		# Dimension variable definitions.
		N, K, M = StateDataSet(X, X0=X0).getDataDimn();

		# Lifted state variable initialization.
		PSI = np.empty( (Nk, K*M) );

		# Lift data set.
		for n in range( K*M ):
			PSI[:,n] = obs( X[:,n,None] )[:,0];

		# Return lifted set.
		return PSI;

class KoopmanOperator(LearningStrategies):
	def __init__(self, obsX, obsY=None, T=None, K=None):
		pass;