import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman');

import numpy as np
import matplotlib.pyplot as plt

# Personal classes.
from KMAN.Operators import *

# Hyper parameter(s).
Nmax = 5;
dN = 1;
beta = 0.01;

# Square wave initialization.
def wave(x):
    return 0.50*np.sin( x ) + 0.15*np.sin( x )**2 + 0.25*np.cos( x )**3;

# Observables
def thetaN(X, N=1):
    k = 0;
    Theta = np.empty( (2*(N+1), X.shape[1]) );
    for i in range( N+1 ):
        Theta[k,:] = np.sin( i*X );
        Theta[k+1,:] = np.cos( i*X );
        k = k + 2;
    return Theta;

if __name__ == '__main__':
    # Generate x-data for square wave.
    T = 10;  Nt = round( T/beta ) + 1;
    X = np.array( [[beta*(i-Nt+1) for i in range( 2*Nt-1 )]] );
    Y = wave( X );

    # Plot results.
    fig, axs = plt.subplots();
    axs.plot( X.T, Y.T, color='r', label='Model' );

    for n in range( 0, Nmax+1, dN ):
        theta = lambda x=None: thetaN( x, N=n );

        solver = Regressor( theta( X ), Y );
        C, _ = solver.ls();

        print( C );
        print( '---------' );

        Yf = C@theta( X );
        axs.plot( X.T, Yf.T, linestyle='-.', label=('N=%i' % n) );

    plt.grid( 1 );
    plt.legend();
    plt.show();
