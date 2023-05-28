import numpy as np
from LearningStrategies import *

# Cascade EDMD
def cascade_edmd(Klist, Tlist, X, Y, X0):
	# If number of operators is 1, solve EDMD
	if len( Klist ) == 1:
		knvar = Klist[0].edmd(X, Y, X0);
		return knvar

	# Otherwise, cut front operator and re-eneter function
	K = cascade_edmd(Klist[1:], Tlist[1:], X, Y, X0);

	# Give resulting operator list to shift function and solve.
	Klist[0].setShiftFunction( Tlist[0](K) );
	Klist[0].edmd(X, Y, X0);

	# Return solved list of Koopman operators.
	return Klist;

# Class: Observables
class Observables:
	def __init__(self, obs):
		# Set obs, Nk and meta variables.
		self.obs = obs;
		self.Nk = obs()['Nk'];
		self.meta = obs();  # optional: for user only

	# Alternative to user called var.obs().
	def lift(self, X):
		return self.obs(X);

	# Assumption: Data set is flat.
	def liftData(self, X):
		# Number of steps and matrix initialization.
		P = len( X[0] );
		Psi = np.empty( (self.Nk, P) );

		# Lift data set.
		for n in range( P ):
			Psi[:,n] = self.obs( X[:,n,None] )[:,0];

		# Return lifted set.
		return Psi;

# Class: KoopmanOperator
class KoopmanOperator:
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
		self.solver = None;
		self.err = -1;

    # Default print function.
	def __str__(self):
		line1 = 'Error: %.5e' % self.err;
		line2 = ', Shape: (' + str(self.K.shape[0]) + ', ' + str(self.K.shape[1]) + ')\n';
		line3 = np.array2string( self.K, precision=5, suppress_small=1 );
		return line1 + line2 + line3;

    # Set shift function post-init.
	def setShiftFunction(self, T):
		self.T = T;
		# Return instance of self.
		return self;

	# Calculate error over data set.
	def resError(self, X, Y, X0=None):
		# Lift sets into observation space.
		PsiX = self.T@self.obsX.liftData(X);
		PsiY = self.obsY.liftData(Y);

		# If solver is not initialized...
		if self.solver is None:
			self.solver = LearningStrategies(PsiX, PsiY);

		# Calculate residual error.
		self.err = self.solver.resError( self.K );

		# Return instance of self.
		return self;

	# Extended Dynamic Mode Decomposition (EDMD)
	def edmd(self, X, Y, X0=None, EPS=None):
		# Lift sets into observation space.
		PsiX = self.T@self.obsX.liftData(X);
		PsiY = self.obsY.liftData(Y);

		# Initialize LearningStrategies class.
		self.solver = LearningStrategies(PsiX, PsiY);

		# Compute Koopman operator through DMD.
		self.K = self.solver.dmd(EPS=EPS);
		self.err = self.solver.resError( self.K );

		# Return instance of self.
		return self;