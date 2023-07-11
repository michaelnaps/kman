import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman');

import numpy as np
import matplotlib.pyplot as plt

# Personal classes.
from KMAN.Regressors import *

# Hyper parameter(s).
Nmax = 500;
dN = 50;
beta = 0.01;

# Square wave initialization.
def wave(x):
    return np.sign( x );

# Observables
def theta(X, N=1):
    k = 0;
    THETA = np.empty( (2*(N+1), X.shape[1]) );
    for i in range( N+1 ):
        THETA[k,:] = np.sin( i*X );
        THETA[k+1,:] = np.cos( i*X );
        k = k + 2;
    return THETA;

if __name__ == '__main__':
    # Generate x-data for square wave.
    T = 1;  Nt = round( T/beta ) + 1;
    X = np.array( [[beta*(i-Nt+1) for i in range( 2*Nt-1 )]] );
    Y = wave( X );

    # Plot results.
    fig, axs = plt.subplots();
    axs.plot( X.T, Y.T, color='r', label='Model' );

    for n in range( 0, Nmax+1, dN ):
        thetaN = lambda x=None: theta( x, N=n );

        solver = Regressor( thetaN( X ), Y );
        C, _ = solver.dmd();

        print( C );
        print( '---------' );

        Yf = C@thetaN( X );
        axs.plot( X.T, Yf.T, linestyle=None, label=('N=%i' % n) );

    plt.grid( 1 );
    plt.legend();
    plt.show();