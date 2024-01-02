import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman')
sys.path.insert(0, expanduser('~')+'/prog/mpc')

import matplotlib.pyplot as plt
from KMAN.Operators import *
from MPC.Optimizer import fdm2c

A = 1.50
p = 0.56
n = 1
m = 10

def polyn(x):
    return x**4 - 3*x**3 + x**2 + x

def model(x):
    alpha = 1e-3
    dx = fdm2c( polyn, x )
    return x - alpha*dx

def obs(x=None):
    if x is None:
        return {'Nk':2}
    m = x.shape[1]
    return np.vstack( (x, np.ones( (1,m) )) )

if __name__ == '__main__':
    # Initial positions.
    X0 = (2*A*np.random.rand( n, m ) - A) + p
    I0 = ['1' if x0 < p else '2' for x0 in X0.T]
    print( 'X0:', X0 )
    print( 'I0:', I0 )

    # Simulation block.
    X1data = []
    X2data = []
    for x0, i0 in zip( X0.T, I0 ):
        x = x0[:,None]
        Xsim = [x]
        xnorm = np.linalg.norm( fdm2c( polyn, x0 ) )
        while xnorm > 1e-3:
            x = model( x )
            xnorm = np.linalg.norm( fdm2c( polyn, x ) )
            Xsim = Xsim + [x]
        if i0 == '1':
            X1data = X1data + [np.hstack( Xsim )]
        else:
            X2data = X2data + [np.hstack( Xsim )]

    # Form training sets.
    X1 = np.hstack( [X[:,:-1] for X in X1data] )
    Y1 = np.hstack( [X[:,1:] for X in X1data] )
    X2 = np.hstack( [X[:,:-1] for X in X2data] )
    Y2 = np.hstack( [X[:,1:] for X in X2data] )

    print( X1 )
    print( Y1 )

    # Kman variables.
    k1var = KoopmanOperator( obs )
    k2var = KoopmanOperator( obs )

    # Solve for Koopman operators on sub-domains.
    k1var.dmd( X1, Y1 )
    k2var.dmd( X2, Y2 )

    print( 'K1:', k1var )
    print( 'K2:', k2var )

    # Plot results.
    fig, axs = plt.subplots()

    # Plot objective over range.
    Xlist = np.linspace( -1.25, 2.5, 1000 )
    Ylist = polyn( Xlist )
    axs.plot( Xlist, Ylist, color='k', linewidth=3 )

    # Plot gradient descent data.
    for X in X1data:
        axs.plot( X.T, polyn( X ).T )
    for X in X2data:
        axs.plot( X.T, polyn( X ).T )

    # Show finished plot.
    axs.grid( 1 )
    # plt.show()
