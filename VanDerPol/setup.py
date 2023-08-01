# Path imports.
import sys
from os.path import expanduser
sys.path.insert( 0, expanduser('~')+'/prog/kman' )

# Standard imports.
import numpy as np
from KMAN.Operators import *

# Hyper parameter(s).
dt = 0.001
Nx = 2

# Duffing model.
def model(x):
    mu = 1
    dx = np.array( [
        x[1],
        mu*(x[1] - x[1]*x[0]**2) - x[0]
    ] )
    return x + dt*dx

# Main execution block.
if __name__ == '__main__':
    # Initial positions.
    A = 5;  N0 = 10
    X0 = 2*A*np.random.rand( Nx, N0 ) - A

    # Generate trajectory data.
    T = 10;  Nt = round( T/dt ) + 1
    tList = np.array( [ [i*dt for i in range( Nt )] ] )
    X = generate_data( tList, model, X0 )