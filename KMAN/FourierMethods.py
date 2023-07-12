# File: LearningStrategies.py
# Created by: Michael Napoli
# Created on: Jul 11, 2023
# Purpose: ...

from KMAN.Regressors import *

class FourierTransform( Regressor ):
    def __init__(self, X, Y, h=1e-3):
        self.X = X;
        self.Y = Y;
        self.N = X.shape[1];
        self.A = np.zeros( (1, self.N) );
        self.B = np.zeros( (1, self.N) );
        self.h = h;
        # Regressor.__init__( self, self.liftData( X ), Y );

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
        return Theta, (xSinList, xCosList);

    def ls(self, EPS=None):
        # Solve using regressor function.
        self.F, self.USV = Regressor.ls( self, EPS=EPS );

        # Return instance of self.
        return self;

    def dft(self):
        self.A[0][0] = 0;
        self.B[0][0] = 1/(2*self.N)*np.sum( self.Y );

        for k in range( 1, self.N-1 ):
            for x, y in zip( self.X[0], self.Y[0] ):
                self.A[0][k] += y*np.sin( 2*np.pi*k*x );
                self.B[0][k] += y*np.cos( 2*np.pi*k*x );

        self.A[0][-1] = 0;
        self.B[0][-1] = 1/(2*self.N)*np.sum( [ self.Y[0][j]*np.cos( 2*np.pi*self.N*self.X[0][j]/self.h ) for j in range( self.N ) ] );

        print( self.A );
        print( self.B );

        # Return instance of self.
        return self;
