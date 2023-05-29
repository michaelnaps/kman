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
		for k, x in enumerate( X.T ):
			Psi[:,k] = self.obs( x[:,None] )[:,0];

		# Return lifted set.
		return Psi;

# Class: Operator
# Purpose: The generalized operator, denoted by C, moves any
#	array, X, into an array, Y. The operator defined here does need
#	to represent a dynamic system, but can be found through the DMD
#	algorithm as long as the appropriate data can be provided.
class Operator:
	def __init__(self, C=None):
		# Shift and Koopman initialization
		self.C = C;

		# Accuracy parameters.
		self.solver = None;
		self.err = -1;

    # Default print function.
	def __str__(self):
		if self.C is None:
			return "\n ERROR: Operator.C has not been set...\n";
		line1 = 'Error: %.5e' % self.err;
		line2 = ', Shape: (' + str(self.K.shape[0]) + ', ' + str(self.K.shape[1]) + ')\n';
		line3 = np.array2string( self.K, precision=5, suppress_small=1 );
		return line1 + line2 + line3;

	# Manually set operator.
	def setOperator(self, C):
		self.C = C;
		# Return instance of self.
		return self;

	# Calculate error over data set.
	# Assumption: C has been set manually or solved for...
	def resError(self, X, Y, X0=None):
		# If solver is not initialized...
		if self.solver is None:
			self.solver = LearningStrategies(X, Y);

		# Calculate residual error.
		self.err = self.solver.resError( self.C );

		# Return instance of self.
		return self;

	# Extended Dynamic Mode Decomposition (EDMD)
	def dmd(self, X, Y, X0=None, EPS=None):
		# Initialize LearningStrategies class.
		self.solver = LearningStrategies(X, Y);

		# Compute Koopman operator through DMD.
		self.C = self.solver.dmd( EPS=EPS );
		self.err = self.solver.resError( self.C );

		# Return instance of self.
		return self;

# Class: KoopmanOperator
# Parent Class: Operator
# Purpose: The operator discussed here defines the propagation of
#	appropriate underlying observation functions forward in time. It
# 	is a more specific case of its parent class, Operator.
class KoopmanOperator( Operator ):
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
			K = np.eye( self.obsY.Nk, self.T.shape[0] );

		# Initialize inhertied class variables.
		Operator.__init__( self, C=K );

	# Class property: K = C (from parent).
	@property
	def K(self):
		# Return the composition operator.
		return self.C;

	def liftData(self, X, Y, X0=None):
		# Lift sets into observation space.
		TPsiX = self.T@self.obsX.liftData( X );
		PsiY  = self.obsY.liftData( Y );
		Psi0  = self.obsX.liftData( X0 );

		# Return lifted data sets.
		return TPsiX, PsiY, Psi0;

	# Redefine residual error from parent for lifted sets.
	def resError(self, X, Y, X0=None):
		# Lift data insto observation space.
		TPsiX, PsiY, Psi0 = self.liftData( X, Y, X0=X0 );

		# Calculate residual error from parent class.
		Operator.resError( self, TPsiX, PsiY, X0=Psi0 );

		# Return instance of self.
		return self;

    # Set shift function post-init.
	def setShiftFunction(self, T):
		self.T = T;
		# Return instance of self.
		return self;

	# Extended Dynamic Mode Decomposition (EDMD)
	def edmd(self, X, Y, X0=None, EPS=None):
		# Lift sets into observation space.
		TPsiX, PsiY, Psi0 = self.liftData( X, Y, X0=X0 );

		# Compute Koopman operator through DMD.
		self.dmd( TPsiX, PsiY, X0=Psi0, EPS=EPS );

		# Return instance of self.
		return self;