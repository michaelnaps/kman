# File: LearningStrategies.py
# Created by: Michael Napoli
# Created on: Jul 11, 2023
# Purpose: ...

from KMAN.Regressors import *

class FourierTransform( Regressor ):
    def __init__(self, X, Y, N=1):
        self.N = N;
        self.F = None;
        self.USV = None;
        Regressor.__init__( self, self.liftData( X ), Y );

    def setLimitNumber( self, N ):
        self.N = N;

        # Return instance of self.
        return self;

    def liftData(self, X):
        M = X.shape[1];
        xSinList = np.empty( (self.N, M) );
        xCosList = np.empty( (self.N, M) );
        for k in range( self.N ):
            xSinList[k,:] = np.sin( k*X );
            xCosList[k,:] = np.cos( k*X );
        Theta = np.vstack( (xSinList, xCosList) );
        return Theta;

    def ls(self, EPS=None):
        # Solve using regressor function.
        self.F, self.USV = Regressor.ls( self, EPS=EPS );

        # Return instance of self.
        return self;
