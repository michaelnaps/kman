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
m = 50      # Number of data points.
b = 2       # Number of fixed points.

def polyn(x):
    return x**4 - 3*x**3 + x**2 + x

def model(x):
    alpha = 1e-3
    dx = fdm2c( polyn, x )
    return x - alpha*dx

def obs(X=None):
    if X is None:
        return {'Nk': 2*n+1}
    psi = np.vstack( (
        X, polyn( X ),
        np.ones( (1,X.shape[1]) )) )
    return psi

def koopmanStack(Klist):
    # Stack operators.
    l = len( Klist )
    p, q = Klist[0].shape

    # Main execution loop.
    Kstack = np.empty( (p*q, l) )
    for k, K in enumerate( Klist ):
        Kstack[:,k] = K.reshape( p*q, )

    # Return grid stack.
    return Kstack

if __name__ == '__main__':
    # Initial positions.
    X0 = (np.linspace( -A, A, m ) + p)[None]
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
    l = 1000  # Number of training points.
    Xstack = np.linspace( -A, A, l )[None] + p
    Kdata = np.array( [Kvarlist[0].K if x < p else Kvarlist[1].K for x in Xstack.T] )
    Kstack = koopmanStack( Kdata )

    # Perform transform.
    Fvar = RealFourier( Xstack, Kstack ).dmd( N=500 )

    # Koopman operator solution and formatting function.
    def koopmanSolve(X):
        Nx = X.shape[1]
        Nk = Kvarlist[0].obsX.Nk
        Klist = Fvar.solve( X ).reshape( Nk,Nk ) if Nx == 1 \
            else Fvar.solve( X ).T.reshape( Nx,Nk,Nk )
        return Klist

    # Plot results.
    fig, axslist = plt.subplots( 2,1 )

    # Plot objective over range.
    xmin = -1.25;  xmax = 2.5
    Xfunc = np.linspace( xmin, xmax, 2*l )
    Yfunc = polyn( Xfunc )
    axslist[0].plot( Xfunc, Yfunc, color='k', linewidth=3 )

    # # Plot gradient descent data.
    # for X in Xdata:
    #     for x in X:
    #         axslist[0].plot( x.T, polyn( x ).T )

    # Operator example cases.
    colorlist = ('cornflowerblue', 'indianred')
    psilist = [
        obs( np.array( [[xmin, p-0.01]] ) ),
        obs( np.array( [[p+0.01, xmax]] ) ) ]
    for j in range( 2500 ):
        for i, Kvar in enumerate( Kvarlist ):
            psilist[i] = Kvar.K@psilist[i]
            if j % 25 == 0:
                axslist[0].plot( psilist[i][0], psilist[i][1],
                    marker='x', markersize=5, linestyle='none',
                    color=colorlist[i] )

    # Split axes into twins to plot together.
    axs1 = axslist[1]
    axs2 = axslist[1].twinx()

    # Coefficient plots.
    step = 1e-3
    Xrange = np.array( [[2*(A*i*step - A/2) + p
        for i in range( round( 1/step ) )]] )
    Krange = koopmanSolve( Xrange )
    axs1.plot( Xrange[0], Krange[:,0,0], color='cornflowerblue' )
    axs2.plot( Xrange[0], Krange[:,0,1], color='indianred', linestyle=':' )

    # Show finished plot.
    for a in axslist:
        a.grid( 1 )
    plt.show()
