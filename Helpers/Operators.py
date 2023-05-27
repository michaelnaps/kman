import numpy as np
from LearningStrategies import *

def cascade_edmd(Klist, Tlist, X, Y, X0):
	pass;

class Observables:
	def __init__(self, obs):
		self.obs = obs;
		self.Nk = obs()['Nk'];
		self.meta = obs();

	# Assumption: Data set is already flattened.
	def liftData(self, X, obs=None):
		# If obs not given, use class.
		if obs is None:
			obs = self.obs;

		# Dimension variable definitions.
		N, K, M = StateDataSet(X).getDataDimn();
		print( N, K, M );
		print( X.shape );

		# Lifted state variable initialization.
		Psi = np.empty( (self.Nk, K*M) );

		# Lift data set.
		print(K*M, X.shape);
		for n in range( K*M ):
			Psi[:,n] = obs( X[:,n,None] )[:,0];

		# Return lifted set.
		return Psi;

class KoopmanOperator(LearningStrategies):
	def __init__(self, obsX, obsY=None, T=None, K=None):
		# Data list variables initially None.
		self.trainingSets = None;

		# X and Y observable initialization.
		self.obsX = Observables( obsX );
		self.obsY = Observables( obsY );

		# Shift function inititalization.
		if T is None:
			self.T = np.eye( self.obsX.Nk );
		else:
			self.T = T;

		# Koopman operator initialization.
		if K is None:
			self.K = np.eye( self.obsY.Nk, self.T.shape[0] );
		else:
			self.K = K;

		# Accuracy parameters.
		self.err = -1;
		self.ind = None;

    # Default print function.
	def __str__(self):
		line1 = 'Error: %.5e' % self.err;
		line2 = ', Shape: (' + str(self.K.shape[0]) + ', ' + str(self.K.shape[1]) + ')\n';
		line3 = np.array2string( self.K, precision=5, suppress_small=1 );
		return line1 + line2 + line3;

	def setTrainingData(self, X, Y, X0=None):
		Psi0 = self.obsX.liftData(X0);
		PsiX = self.obsX.liftData(X);
		PsiY = self.obsY.liftData(Y);
		self.trainingData = LearningStrategies(PsiX, PsiY, X0=Psi0);
		return self;

    # Set shift function post-init.
	def setShiftFunction(self, T):
		self.T = T;
		return self;