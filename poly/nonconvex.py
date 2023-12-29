import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman')
sys.path.insert(0, expanduser('~')+'/prog/mpc')

import matplotlib.pyplot as plt
from KMAN.Operators import *
from MPC.Optimizer import fdm2c

n = 1
m = 10
A = 0.001

def polyn(x):
    return x**4 - 3*x**3 + x**2 + x

def model(x):
    alpha = 1e-3
    dx = fdm2c( polyn, x )
    return x - alpha*dx

if __name__ == '__main__':
    # Initial positions.
    X0 = (2*A*np.random.rand( n, m ) - A) + 0.56
    print( 'X0:', X0 )

    # Simulation through initial positions.
    Xdata = []
    for x0 in X0.T:
        x = x0[:,None]
        Xsim = [x]
        xnorm = np.linalg.norm( fdm2c( polyn, x0 ) )
        while xnorm > 1e-3:
            x = model( x )
            xnorm = np.linalg.norm( fdm2c( polyn, x ) )
            Xsim = Xsim + [x]
        Xdata = Xdata + [np.hstack( Xsim )]

    # Plot results.
    fig, axs = plt.subplots()
    axs.grid( 1 )

    # Plot objective over range.
    Xlist = np.linspace( -1.25, 2.5, 1000 )
    Ylist = polyn( Xlist )
    axs.plot( Xlist, Ylist, color='k', linewidth=3 )

    # Plot gradient descent data.
    for X in Xdata:
        axs.plot( X.T, polyn( X ).T )

    # Show finished plot.
    plt.show()
