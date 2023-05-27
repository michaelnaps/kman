# File: LearningStrategies.py
# Created by: Michael Napoli
# Created on: May 26, 2023
# Purpose: To create a class which executes various learning strategies over sets
# 	of state propagation data defined and inherited from the StateDataSet class.

import numpy as np

# Class: DataSet
# Assumption: Uniform series/stacked data structure.
#	If multiple
class StateDataSet:
	def __init__(self, X, X0=None):
		self.setNewData(X, X0=X0);

	def setNewData(self, X, X0=None):
		self.X = X;

		if X0 is None:
			N, K = X.shape;
			M = 1;
		else:
			N = len(X0);		# Number of state variables
			K = len(X[0]);		# Number of time-steps
			M = len(X0[0]);		# Number of data sets


	def stack_data(self):
		xStack = np.empty( (N, M*K) );

		n = 0;
		k = 0;
		for i in range(M):
			xStack[:,k:k+K] = self.X[n:n+N,:];
			n += N;
			k += K;

		self.setNewData(xStack, X0=x0);
		return self;

# Class: LearningStrategies
class LearningStrategies:
	def __init__(self, X, Y, X0=None, Y0=None):
		if Y0 is None:
			Y0 = X0;
		self.Xset = StateDataSet(X, X0=X0);
		self.Yset = StateDataSet(Y, X0=Y0);