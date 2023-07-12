import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman');

import numpy as np
import matplotlib.pyplot as plt

# Personal classes.
from KMAN.Regressors import *
from KMAN.FourierMethods import *

# Hyper parameter(s).
beta = 0.01;

# Square wave initialization.
def wave(x):
    return np.sign( x );

if __name__ == '__main__':
    # Generate x-data for square wave.
    T = 1;  Nt = round( T/beta ) + 1;
    X = np.array( [[beta*(i-Nt+1) for i in range( 2*Nt-1 )]] );
    Y = wave( X );

    # Test Fourier method class.
    fvar = FourierTransform( X, Y );
    fvar.dft();

    # Results data
    print( fvar.A.shape );
    print( fvar.B.shape );
    xSinList, xCosList = fvar.liftData( X )[1];
    Yf = fvar.A@xSinList+ fvar.B@xCosList;

    print( X );
    print( Y );
    print( Yf );

    # Plot results
    fig, axs = plt.subplots();
    axs.plot( X.T, Y.T );
    axs.plot( X.T, Yf.T );
    axs.grid( 1 );
    plt.show();