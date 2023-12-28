import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman')
sys.path.insert(0, expanduser('~')+'/prog/mpc')

from KMAN.Operators import *
from MPC.Optimizer import fdm2c

def polyn(x):
    return x**4 - 3*x**3 + x**2 + x

def model(x):
    alpha = 1e-3
    dx = fdm2c( polyn, x )
    return x - alpha*dx

if __name__ == '__main__':
    x = np.array( [[0]] )
    dx = 1
    while np.linalg.norm( dx ) > 1e-6:
        x = model( x )
        dx = fdm2c( polyn, x )
        print( x )
