# File: LearningStrategies.py
# Created by: Michael Napoli
# Created on: Jul 11, 2023
# Purpose: ...

from KMAN.Regressors import *

class Fourier( Regressor ):
    def __init__(self, N=1, X=None, Y=None, X0=None, Y0=None):
        self.N = N;
        Regressor.__init__( self, X=X, Y=Y, X0=X0, Y0=Y0 );

    def least_squares
