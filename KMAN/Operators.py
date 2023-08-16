import numpy as np
from KMAN.Regressors import *

# Cascade EDMD
# Inputs:
#	Tlist: 	tuple of shift functions.
#	Klist: 	tuple of Koopman operators.
# 	X: 		tuple of data sets.
#	Y: 		tuple of data sets (propagation of X).
#	X0: 	array of initial positions. (optional)
def cascade_edmd(Tlist, Klist, X, Y, X0=None):
	# If number of operators is 1, solve EDMD.
	if len( Klist ) == 1:
		Klist[0].edmd( X[0], Y[0], X0=X0 )
		return (Klist,)

	# Otherwise, cut front operator and re-eneter function.
	cascade_edmd( Tlist[1:], Klist[1:], X[1:], Y[1:], X0 )

	# Solve for shift function and compute operator.
	if Tlist[0] is not None:
		Klist[0].setShiftFunction( Tlist[0]( Klist[1:] ) )
	Klist[0].edmd( X[0], Y[0], X0 )

	# Return solved list of Koopman operators.
	return Klist

# Class: Observables
class Observables:
	def __init__(self, obs):
		# Set obs, Nk and meta variables.
		self.obs = obs
		self.Nk = obs()['Nk']
		self.meta = obs()  # optional: for user only

	# Alternative to user called var.obs().
	def lift(self, X):
		return self.obs( X )

	# Assumption: Data set is flat.
	def liftData(self, X):
		# For rare case where X is None
		if X is None:
			return None

		# Number of steps and matrix initialization.
		P = X.shape[1]
		Psi = np.empty( (self.Nk, P) )

		# Lift data set.
		for k, x in enumerate( X.T ):
			Psi[:,k] = self.obs( x[:,None] )[:,0]

		# Return lifted set.
		return Psi

# Class: Operator
# Purpose: The generalized operator, denoted by C, moves any
#	array, X, into an array, Y. The operator defined here does need
#	to represent a dynamic system, but can be found through the DMD
#	algorithm as long as the appropriate data can be provided.
class Operator:
	def __init__(self, C=None):
		# Shift and Koopman initialization
		self.C = C
		self.USV = None

		# Accuracy parameters.
		self.solver = None
		self.err = -1

    # Default print function.
	def __str__(self):
		if self.C is None:
			return "\n ERROR: Operator.C has not been set...\n"
		line1 = 'Error: %.5e' % self.err
		line2 = ', Shape: (' + str(self.K.shape[0]) + ', ' + str(self.K.shape[1]) + ')\n'
		line3 = np.array2string( self.K, precision=5, suppress_small=1 )
		return line1 + line2 + line3

	# Manually set operator.
	def setOperator(self, C):
		self.C = C
		# Return instance of self.
		return self

	# Calculate error over data set.
	# Assumption: C has been set manually or solved for...
	def resError(self, X, Y, X0=None, C=None, save=0):
		# Temporary solver for calculating error.
		solver = Regressor(X, Y)

		# In the event an alternative operator should be tested.
		if C is None:
			C = self.C

		# Calculate and return residual error.
		err = solver.resError( C )

		# Save if requested and return.
		if save:
			self.err = err
		return err

	# Extended Dynamic Mode Decomposition (EDMD)
	def dmd(self, X, Y, X0=None, EPS=None):
		# Initialize Regressor class.
		self.solver = Regressor(X, Y)

		# Compute Koopman operator through DMD.
		self.C, self.USV = self.solver.dmd( EPS=EPS )
		self.err = self.solver.resError( self.C )

		# Return instance of self.
		return self

# Class: KoopmanOperator
# Parent Class: Operator
# Purpose: The Koopman operator is defined as an operator matrix
#	which takes a list of linear/nonlinear observation functions
# 	and propagates them forward w.r.t to a constant time-step.
# Principle equation: Psi(x+) = K Psi(x)
class KoopmanOperator( Operator ):
	def __init__(self, obsX, obsY=None, T=None, K=None):
		# Data list variables initially None.
		self.trainingSets = None

		# X and Y observable initialization.
		if obsY is None:
			obsY = obsX
		self.obsX = Observables( obsX )
		self.obsY = Observables( obsY )

		# Shift function inititalization.
		if T is None:
			self.T = np.eye( self.obsX.Nk )
		else:
			self.T = T

		# Koopman operator initialization.
		if K is None:
			K = np.eye( self.obsY.Nk, self.T.shape[0] )

		# Initialize inhertied class variables.
		Operator.__init__( self, C=K )

	# Class property: K = C (from parent).
	@property
	def K(self):
		# Return the composition operator.
		return self.C

    # Set shift function post-init.
	def setShiftFunction(self, T):
		self.T = T
		# Return instance of self.
		return self

	def liftData(self, X, Y, X0=None):
		# Lift sets into observation space.
		TPsiX = self.T@self.obsX.liftData( X )
		PsiY  = self.obsY.liftData( Y )
		Psi0  = self.obsX.liftData( X0 )

		# Return lifted data sets.
		return TPsiX, PsiY, Psi0

	def propagate(self, X):
		if self.K is None:
			print( "\nERROR: Operator is unset...\n" )
			return None
		return self.K@self.obsX.liftData( X )

	# Redefine residual error from parent for lifted sets.
	def resError(self, X, Y, X0=None, save=0):
		# Lift data insto observation space.
		TPsiX, PsiY, Psi0 = self.liftData( X, Y, X0=X0 )

		# Calculate residual error from parent class.
		err = Operator.resError( self, TPsiX, PsiY, X0=Psi0, save=save )

		# Return instance of self.
		return err

	# Extended Dynamic Mode Decomposition (EDMD)
	def edmd(self, X, Y, X0=None, EPS=None):
		# Lift sets into observation space.
		TPsiX, PsiY, Psi0 = self.liftData( X, Y, X0=X0 )

		# Compute Koopman operator through DMD.
		self.dmd( TPsiX, PsiY, X0=Psi0, EPS=EPS )

		# Return instance of self.
		return self

	# Convert Lie operator to discrete Koopman operator with time-step.
	# Assumption(s):
	#	1). Lie operator is diagonalizable.
	#	2). Eigenvectors of L create an invertible matrix.
	def L2K(self, lvar, dt=1e-3):
		# Grab eignvalues and invert for transition.
		_, V = np.linalg.eig( lvar.L )
		Vinv = np.linalg.solve( V,np.eye( self.obsX.Nk ) )

		# Calculate the logarithm of K.
		Lp = Vinv@lvar.L@V
		eLp = np.diag( np.exp( np.diag( dt*Lp ) ) )
		eL = V@eLp@Vinv

		# Calculate the Lie operator.
		self.C = eL

		# Return instance of self.
		return self

# Class: LieOperator
# Parent Class: KoopmanOperator
# Grandparent Class: Operator
# Purpose: The operator defined in this class is the continuous-time
#	analog to the Koopman opreator. In other words, it uses the
#	appropriately defined observation functions to calculate the rate-
#	of-change of the entire list as opposed to taking discrete steps.
# Principle equation: (d/dt) Psi(x) = L Psi(x)
class LieOperator( KoopmanOperator ):
	def __init__(self, obsX, obsY=None, T=None, L=None):
		KoopmanOperator.__init__( self, obsX, obsY=obsY, T=T, K=L )

	@property
	def L(self):
		return self.C

	# Elementary IVP solution method.
	# Assumption: dt << 1, or the system is strongly linear.
	def propagate(self, X, dt=1e-3):
		PsiX = self.obsX.lift( X )
		dPsiX = KoopmanOperator.propagate( self, X )
		return PsiX + dt*dPsiX

	# Calculate the residual error of the operator compared to
	#	some given data set, X and Y.
	# Note: Uses the class elementary IVP solver.
	def resError(self, X, Y, dt=1e-3, save=0):
		# Lift the Y data set
		PsiY = self.obsY.liftData( Y )

		# Propagate the X set forward using L
		PsiXp = self.propagate( X, dt=dt )

		# Identity matrix used as operator since propagating in function.
		I = np.eye( self.obsY.Nk, self.obsX.Nk )

		# Calculate the residual error
		Operator.resError( self, PsiY, PsiXp, C=I, save=save )

		# Return instance of self.
		return self

	# Convert Koopman operator with time-step to Lie operator.
	# Assumption(s):
	#	1). Koopman operator is diagonalizable.
	#	2). Eigenvectors of K create an invertible matrix.
	def K2L(self, kvar, dt=1e-3):
		# Grab eignvalues and invert for transition.
		_, V = np.linalg.eig( kvar.K )
		Vinv = np.linalg.solve( V,np.eye( self.obsX.Nk ) )

		# Calculate the logarithm of K.
		Kp = Vinv@kvar.K@V
		logKp = np.diag( np.log( np.diag( Kp ) ) )
		logK = V@logKp@Vinv

		# Calculate the Lie operator.
		self.C = 1/dt*logK

		# Return instance of self.
		return self