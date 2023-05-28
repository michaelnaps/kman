# File: LearningStrategies.py
# Created by: Michael Napoli
# Created on: May 26, 2023
# Purpose: To create a class which executes various learning strategies over sets
# 	of state propagation data defined and inherited from the StateDataSet class.

import numpy as np

# Helper data function for generating sets.
def generate_data(tlist, model, X0, control=None, Nu=0):
    Nx = len(X0);
    N0 = len(X0[0]);
    Nt = len(tlist[0]);

    if control is not None:
        ulist = np.empty( (N0*Nu, Nt-1) );
    else:
        ulist = None;
        Nu = Nx;

    xlist = np.empty( (N0*Nx, Nt) );

    n = 0;
    k = 0;
    for i in range(N0):

        if control is not None:
            u = np.empty( (Nu, Nt-1) );
        x = np.empty( (Nx, Nt) );

        x[:,0] = X0[:,i];

        for t in range(Nt-1):
            if control is not None:
                u[:,t] = control(x[:Nx,t]).reshape(Nu,);
                x[:,t+1] = model(x[:Nx,t], u[:,t]).reshape(Nx,);
            else:
                x[:,t+1] = model(x[:Nx,t]).reshape(Nx,);

        if control is not None:
            ulist[k:k+Nu,:] = u;
            k = k + Nu;

        xlist[n:n+Nx,:] = x;
        n = n + Nx;

    return xlist, ulist;

# Class: DataSet
# Assumption: Uniform series/stacked data structure.
class StateDataSet:
	def __init__(self, X, X0=None):
		self.setNewData(X, X0=X0);

	def setNewData(self, X, X0=None):
		self.X = X;
		self.X0 = X0;0

		# If X0 is None, then data is flat.
		if X0 is None:
			self.N, self.P = X.shape;
			self.M = 1;
		# otherwise, data has 3-D shape (M).
		else:
			self.N = len( X0 );			# Number of states
			self.P = len( X[0] );		# Number of time-steps
			self.M = len( X0[0] );		# Number of data sets

	def getDataDimn(self):
		return self.N, self.P, self.M;

	def flattenData(self, suppress=0):
		# Execute flatten if M > 1
		if self.M > 1:
			# Warning.
			if not suppress:
				print( "\nWARING: Flatten function cannot be undone...\n")

			# Initialize flattened data matrix.
			Xflat = np.empty( (self.N, self.M*self.P) );

			# Iterate through and reshape matrix.
			n = 0;
			p = 0;
			for i in range(self.M):
				Xflat[:,p:p+self.P] = self.X[n:n+self.N,:];
				n += self.N;
				p += self.P;

			# Set internal data to flattened data.
			self.setNewData(Xflat, X0=None);
		# otherwise,
		elif not suppress:
			# print warning.
			print( "\nWARNING: Data already flattened...\n" )

		# Return instance of self.
		return self;

# Class: LearningStrategies
# Assumptions: Y set is a forward snapshot of X; meaning
#	they share a common length, even if the number of states
#	represented (width) is different.
#	In other words, the statement F: X -> Y is justifiable by
#	the learning stategies presented.
class LearningStrategies:
	def __init__(self, X, Y, X0=None, Y0=None):
		self.setDataLists(X, Y, X0=X0, Y0=Y0);
		self.err = -1;

	# Set data variables post-init.
	def setDataLists(self, X, Y, X0=None, Y0=None):
		# If Y0 is None set equal to X0.
		if Y0 is None:
			Y0 = X0;

		# Create two snapshot data sets
		self.Xset = StateDataSet(X, X0=X0);
		self.Yset = StateDataSet(Y, X0=Y0);

		# Flatten data (suppress warning).
		self.Xset.flattenData(suppress=1);
		self.Yset.flattenData(suppress=1);
		return self;

	# Calculate the residual error of a given operator.
	def resError(self, C):
		err = 0;
		for n in range( self.Xset.P ):
			err += np.linalg.norm(self.Yset.X[:,n,None] - C@self.Xset.X[:,n,None])**2;
		return err;

	# Dynamic Mode Decomposition (DMD)
	# Assumption: Datasets are already flattened.
	def dmd(self, EPS=None):
		# Get set dimensions.
		# Assume Yset is similarly shaped.
		TOL = 1E-12;
		_, K, _ = self.Xset.getDataDimn();

		# Perform DMD on data sets.
		# Create least-squares regression matrices for
		# 	F = inv(G)*A,
		# where O is operator of interest.
		G = 1/K * (self.Xset.X @ self.Xset.X.T);
		A = 1/K * (self.Xset.X @ self.Yset.X.T);

		# Get single value decomposition (SVD) matrices.
		(U, S, V) = np.linalg.svd(G);

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
		C = A.T @ (U @ Sinv @ V.T);

		self.err = self.resError(C);
		return C;
