import numpy as np
from LearningStrategies import *

def cascade_edmd(Klist, Tlist, X, Y, X0):
	pass;

class Observables:
	def __init__(self, obs):
		# Set obs, Nk and meta variables.
		self.obs = obs;
		self.Nk = obs()['Nk'];
		self.meta = obs();  # optional: for user only

	# Assumption: Data set is flat.
	def liftData(self, X):

		# Number of steps and matrix initialization.
		P = len( X[0] );
		Psi = np.empty( (self.Nk, P) );

		# Lift data set.
		for n in range( P ):
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

	def initTrainingData(self, X, Y, X0=None):
		# Lift sets into observation space.
		PsiX = self.obsX.liftData(X);
		PsiY = self.obsY.liftData(Y);
		PsiX0 = self.obsX.liftData(X0);

		# Initialize LearningStrategies class.
		LearningStrategies(self, PsiX, PsiY, X0=PsiX0);

		# Return instance of self.
		return self;

    # Set shift function post-init.
	def setShiftFunction(self, T):
		self.T = T;
		# Return instance of self.
		return self;

	# Extended Dynamic Mode Decomposition (EDMD)
	def edmd(self, X, Y, X0=None, EPS=None):
		# Lift data sets into observation space.
		self.liftDataSets(X, Y, X0=X0);

		# Compute Koopman operator through DMD.
		self.K = self.dmd(EPS=EPS);

		# Return instance of self.
		return self;