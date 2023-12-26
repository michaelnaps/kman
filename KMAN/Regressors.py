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
		# Number of points in data.
		P = self.Xset.P

		# Calculate residual error in between sets.
		err = 0
		for n in range( self.Xset.P ):
			err += np.linalg.norm( self.Yset.X[:,n,None] - C@self.Xset.X[:,n,None] )**2

		# Return residual error.
		return 1/P*err

	# Dynamic Mode Decomposition (DMD).
	# Assumption: Datasets are already flattened.
	def dmd(self, EPS=None):
		# Get set dimensions.
		# Assume Yset is similarly shaped.
		TOL = 1E-12
		_, K, _ = self.Xset.getDataDimn()

		# Perform DMD on data sets.
		# Create least-squares regression matrices for
		# 	C = inv(G)*A,
		# where C is operator of interest.
		G = 1/K * (self.Xset.X @ self.Xset.X.T)
		A = 1/K * (self.Xset.X @ self.Yset.X.T)

		# Get single value decomposition (SVD) matrices.
		(U, S, V) = np.linalg.svd( G )

		# Get priority functions from S.
		EPS = TOL*max( S ) if EPS is None else EPS
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

	# Classic Proper Orthonal Decomposition (CPOD)
	# Assumption: Datasets are already flattened.
	# 			  X and Y are sequentially ordered sets.
	def cpod(self, m=None, EPS=1e-21):
		# Initialize coefficient matrix.
		N, P, _ = self.Xset.getDataDimn()
		M = N if m is None else m
		A = np.empty( (N, P+1) )

		# Combine X and Y sets and remove spatial mean.
		Q = np.hstack( (self.Xset.X, self.Yset.X[:,-1,None]) )
		qAvg = np.mean( Q, axis=1 )[:,None]
		X = Q - qAvg

		# Calculate covariance matrix and derive eigenvalues.
		R = 1/(P + 1)*X@X.T

		# Iterate through X and solve linear SOE.
		H = np.empty( (N, P+1) )
		for i, x in enumerate( X.T ):
			h = np.linalg.solve( R, x[:,None] )
			H[:,i] = h[:,0]

		print( H )
		print( np.linalg.norm( Q - (H@X + qAvg) ) )

		return