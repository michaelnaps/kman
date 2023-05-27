# File: LearningStrategies.py
# Created by: Michael Napoli
# Created on: May 26, 2023
# Purpose: To create a class which executes various learning strategies over sets
# 	of data defined and inherited from the DataSet class.

import numpy as np

# Class: DataSet
class DataSet:
	def __init__(self, X, Y, X0=None):
		self.X = X;
		self.Y = Y;

# Class: LearningStrategies
class LearningStrategies(DataSet):
	def __init__(self, X, Y, X0=None):
		DataSet.__init__(self, X, Y, X0=X0);
