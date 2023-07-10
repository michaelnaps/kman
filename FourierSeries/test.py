import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman');

import numpy as np
import matplotlib.pyplot as plt

# Personal classes.
from KMAN.Operators import *

# Hyper parameter(s).
N = 100;
beta = 0.001;

# Square wave initialization.
def wave(x):
    return np.sign( x );

# Observables
def obsX(x=None):
    if x is None:
        meta = {'Nk':N};
        return meta;

    Psi = np.empty( (N,1) );
    for i in range( N ):
        Psi[i] = np.sin( i*x );

    return Psi;

def obsY(x=None):
    if x is None:
        meta = {'Nk':1};
        return meta;
    return x;

if __name__ == '__main__':
    # Generate x-data for square wave.
    T = 1;  Nt = round( T/beta ) + 1;
    X = np.array( [[beta*(i-Nt) for i in range( 2*Nt )]] );
    Y = np.sign( X );

    # Solve using Koopman operator class.
    kvar = KoopmanOperator( obsX, obsY=obsY );
    kvar.edmd( X, Y );

    # For final plot.
    Ytest = kvar.propagate( X );
    print( Ytest );

    # Plot results.
    fig, axs = plt.subplots();
    axs.plot( X.T, Y.T, label='True' );
    axs.plot( X.T, Ytest.T, label='Fourier' );
    plt.show();
