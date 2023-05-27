# File: LearningStrategies.py
# Created by: Michael Napoli
# Created on: May 26, 2023
# Purpose: To create a class which executes various learning strategies over sets
# 	of state propagation data defined and inherited from the StateDataSet class.

import numpy as np

# Class: DataSet
# Assumption: Uniform series/stacked data structure.
class StateDataSet:
	def __init__(self, X, X0=None):
		self.setNewData(X, X0=X0);

	def setNewData(self, X, X0=None):
		self.X = X;

		# If X0 is None, then data is flat.
		if X0 is None:
			self.N, self.K = X.shape;
			self.M = 1;
		# otherwise, data has 3-D shape (M).
		else:
			self.N = len(X0);		# Number of states
			self.K = len(X[0]);		# Number of time-steps
			self.M = len(X0[0]);	# Number of data sets

	def getDataDimn(self):
		return self.N, self.K, self.M;

	def flattenData(self, suppress=0):
		# Execute flatten if M > 1
		if self.M > 1:
			Xflat = np.empty( (self.N, self.M*self.K) );

			n = 0;
			k = 0;
			for i in range(self.M):
				Xflat[:,k:k+self.K] = self.X[n:n+self.N,:];
				n += self.N;
				k += self.K;

			self.setNewData(Xflat, X0=None);
		# otherwise, print warning.
		elif not suppress:
			print("\nWARNING: Data already flattened...\n")

		return self;

# Class: LearningStrategies
# Assumptions: Y set is a forward snapshot of X; meaning
#	they share a common length, even if the number of states
#	represented (width) is different.
#	I.e. the statement F: X -> Y is justifiable by the
#	learning stategies presented.
class LearningStrategies:
	def __init__(self, X, Y, X0=None, Y0=None):
		# If Y0 is None set equal to X0.
		if Y0 is None:
			Y0 = X0;

		# Create two snapshot data sets
		self.Xset = StateDataSet(X, X0=X0);
		self.Yset = StateDataSet(Y, X0=Y0);

		# Flatten data (suppress warning).
		self.Xset.flattenData(suppress=1);
		self.Yset.flattenData(suppress=1);

	# Dynamic Mode Decomposition (DMD)
	# Assumption: Datasets are already flattened.
	def dmd(self, EPS=None):
		# Get set dimensions.
		# Assume Yset is similarly shaped.
		_, K, _ = self.Xset.getDataDimn();

		# Perform DMD on data sets.
		# Create least-squares regression matrices for
		# 	F = inv(G)*A,
		# where O is operator of interest.
		G = 1/K * (self.Xset.X @ self.Xset.X.T);
		A = 1/K * (self.Xset.X @ self.Yset.X.T);

		# Get single value decomposition (SVD) matrices.
		(U, S, V) = mp.linalg.svd(G);

		# Get priority functions from S.
		if EPS is None:
			EPS = TOL*max(S);
		ind = S > EPS;

		# Truncate space for prioritized functions.
		U = U[:,ind];
		S = S[ind];
		V = V[ind,:].T;

		# Invert S values and create matrix.
		Sinv = np.diag([1/S[i] for i in range(len(S))]);

		# Solve for the DMD operator (TRANSPOSED) and return.
		F = A.T @ (U @ Sinv @ V.T);
		return F;
