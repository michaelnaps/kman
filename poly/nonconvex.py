import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman')
sys.path.insert(0, expanduser('~')+'/prog/four')
sys.path.insert(0, expanduser('~')+'/prog/mpc')

import matplotlib.pyplot as plt
from KMAN.Operators import *
from FOUR.Transforms import *
from MPC.Optimizer import fdm2c

p = 0.56    # Center of fixed point boundary.
A = 2.00    # Width of random initial position.
n = 1       # Dimension of x/f(x).
m = 10      # Number of data points.
b = 2       # Number of fixed points.

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
    psi = np.vstack( (x, np.ones( (1,m) )) )
    return psi

def koopmanStack(Klist):
    # Stack operators.
    p = len( Klist )
    n, m = Klist[0].shape

    # Main execution loop.
    Kstack = np.empty( (n*m, p) )
    for k, K in enumerate( Klist ):
        Kstack[:,k] = K.reshape( n*m, )

    # Return grid stack.
    return Kstack

if __name__ == '__main__':
    # Initial positions.
    X0 = (2*A*np.random.rand( n, m ) - A) + p
    I0 = [0 if x0 < p else 1 for x0 in X0.T]
    # print( 'X0:', X0 )
    # print( 'I0:', I0 )

    # Simulation block.
    Xdata = [[] for _ in range( b )]
    for x0, i0 in zip( X0.T, I0 ):
        x = x0[:,None]
        Xsim = [x]
        xnorm = np.linalg.norm( fdm2c( polyn, x0 ) )
        while xnorm > 1e-3:
            x = model( x )
            xnorm = np.linalg.norm( fdm2c( polyn, x ) )
            Xsim = Xsim + [x]
        Xdata[i0] = Xdata[i0] + [np.hstack( Xsim )]

    # Form training sets.
    Xlist = [ np.hstack( [x[:,:-1] for x in X] )
        for X in Xdata ]
    Ylist = [ np.hstack( [x[:,1:] for x in X] )
        for X in Xdata ]

    # Kman variables.
    Kvarlist = [KoopmanOperator( obs ).edmd( X, Y )
        for X, Y in zip( Xlist, Ylist ) ]

    # Print operators.
    print( 'Sub-domain Operators:' )
    for i, K in enumerate( Kvarlist ):
        print( 'K%s:' % (i + 1), K )

    # Format operator list in prep for Fourier transform.
    Kdata = [[Kvar.K for x in X.T] for Kvar, X in zip( Kvarlist, Xlist )]
    Ktemp = []
    for i, Klist in enumerate( Kdata ):
        Ktemp = Ktemp + [koopmanStack( Klist )]
    Xstack = np.hstack( Xlist )
    Kstack = np.hstack( Ktemp )

    # Perform transform.
    Fvar = RealFourier( Xstack, Kstack ).dmd( N=100 )

    # Koopman operator solution and formatting function.
    def koopmanSolve(X):
        Nx = X.shape[1]
        Nk = Kvarlist[0].obsX.Nk
        Klist = Fvar.solve( X ).reshape( Nk,Nk ) if Nx == 1 \
            else Fvar.solve( X ).T.reshape( Nx,Nk,Nk )
        return Klist

    # Plot results.
    fig, axs = plt.subplots()

    # Plot objective over range.
    Xfunc = np.linspace( -1.25, 2.5, 1000 )
    Yfunc = polyn( Xfunc )
    axs.plot( Xfunc, Yfunc, color='k', linewidth=3 )

    # Plot gradient descent data.
    for X in Xdata:
        for x in X:
            axs.plot( x.T, polyn( x ).T )

    # Operator example cases.
    colorlist = ('cornflowerblue', 'indianred')
    psilist = [
        obs( np.array( [[-1.25, p-0.01]] ) ),
        obs( np.array( [[p+0.01, 2.500]] ) ) ]
    for j in range( 2500 ):
        for i, Kvar in enumerate( Kvarlist ):
            psilist[i] = Kvar.K@psilist[i]
            if j % 50 == 0:
                axs.plot( psilist[i][0], polyn( psilist[i][0] ),
                    marker='x', markersize=5,
                    linestyle='none', color=colorlist[i] )

    # Show finished plot.
    axs.grid( 1 )
    plt.show()
