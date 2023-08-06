# File: Regressors.py
# Created by: Michael Napoli
# Created on: May 26, 2023
# Purpose: To create a class which executes various learning strategies over sets
# 	of state propagation data defined and inherited from the DataSet class.

import numpy as np
from KMAN.DataSets import *

# Class: LearningStrategies
# Assumptions: Y set is a forward snapshot of X meaning
#	they share a common width (m), even if the number of states
#	represented (ny vs. nx) is different. The intent of this
#	class is to solve for an operator C which accurately moves
#	between these two sets. In other words, the statement
#		F: X -> Y
#	is justifiable by the operator
#		Y = C X.
#	Where the height of C is equal to (ny) and the width of C
#	is equal to (nx).
class Regressor:
	def __init__(self, X, Y, X0=None, Y0=None):
		self.setDataLists( X, Y, X0=X0, Y0=Y0 )

	# Set data variables post-init.
	def setDataLists(self, X, Y, X0=None, Y0=None):
		# If Y0 is None set equal to X0.
		if Y0 is None:
			Y0 = X0

		# Create two snapshot data sets
		self.Xset = DataSet( X, X0=X0 )
		self.Yset = DataSet( Y, X0=Y0 )

		# Flatten data (suppress warning).
		self.Xset.flattenData( verbose=0 )
		self.Yset.flattenData( verbose=0 )
		return self

	# Calculate the residual error of a given operator.
	def resError(self, C):
		# Calculate residual error in between sets.
		err = 0
		for n in range( self.Xset.P ):
			err += np.linalg.norm( self.Yset.X[:,n,None] - C@self.Xset.X[:,n,None] )**2

		# Return residual error.
		return err

	# Least Squares (LS)
	# Assumption: Datasets are already flattened.
	def ls(self, EPS=None):
		# Get set dimensions.
		# Assume Yset is similarly shaped.
		TOL = 1E-12
		_, K, _ = self.Xset.getDataDimn()

		# Perform DMD on data sets.
		# Create least-squares regression matrices for
		# 	F = inv(G)*A,
		# where O is operator of interest.
		G = 1/K * (self.Xset.X @ self.Xset.X.T)
		A = 1/K * (self.Xset.X @ self.Yset.X.T)

		# Get single value decomposition (SVD) matrices.
		(U, S, V) = np.linalg.svd( G )

		# Get priority functions from S.
		if EPS is None:
			EPS = TOL*max( S )
		ind = S > EPS

		# Truncate space for prioritized functions.
		Ut = U[:,ind]
		St = S[ind]
		Vt = V[ind,:].T

		# Invert S values and create matrix.
		Sinv = np.diag( [1/St[i] for i in range( len( St ) )] )

		# Solve for the DMD operator (TRANSPOSED) and return.
		C = A.T @ (Ut @ Sinv @ Vt.T)

		self.err = self.resError( C )
		return C, (U,S,V)

	# Proper Orthonal Decomposition (POD)
	def pod(self):
		pass