import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman');

import numpy as np
import matplotlib.pyplot as plt

# Personal classes.
from KMAN.Operators import *

# Hyper parameter(s).
N = 5;
# N = 1000;
# N = 10000;  # Near perfect...
beta = 0.01;

# Square wave initialization.
def wave(x):
    return 0.50*np.sin( x ) + 0.15*np.sin( x )**2 + 0.25*np.cos( x )**3;

# Observables
def obsX(x=None):
    if x is None:
        meta = {'Nk':2*N};
        return meta;

    k = 0;
    Psi = np.empty( (2*N,1) );
    for i in range( N ):
        Psi[k] = np.sin( i*x );
        Psi[k+1] = np.cos( i*x );
        k = k + 2;

    return Psi;

def obsY(x=None):
    if x is None:
        meta = {'Nk':1};
        return meta;
    return x;

if __name__ == '__main__':
    # Generate x-data for square wave.
    T = 10;  Nt = round( T/beta ) + 1;
    X = np.array( [[beta*(i-Nt) for i in range( 2*Nt )]] );
    Y = wave( X );

    # Solve using Koopman operator class.
    kvar = KoopmanOperator( obsX, obsY=obsY );
    print( kvar.edmd( X, Y ) );

    # For final plot.
    Ytest = kvar.propagate( X );

    # Plot results.
    fig, axs = plt.subplots();
    axs.plot( X.T, Y.T, label='True' );
    axs.plot( X.T, Ytest[0,:].T, linestyle='--', label='Fourier' );
    plt.show();
