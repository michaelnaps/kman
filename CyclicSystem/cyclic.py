import sys
sys.path.insert(0, '../');

from root import *

# observable identification: control
def obs(X=None):
    if X is None:
        meta = {'Nk':Nx+(Nu-1)+3};
        return meta;

    x = X[:Nx];
    u = X[Nx:Nx+Nu-1];

    xTrig = np.array( [np.sin( x[2] ), np.cos( x[2] )] );
    Psi = np.vstack( (x, u, xTrig, [1]) );

    return Psi;

# Main execution block.
if __name__ == '__main__':
    x0 = np.array( [[R],[0],[0]] );
    xu0 = np.vstack( (x0, [[0],[0],[0]]) );

    # simulation time variables
    T = 10;  Nt = round( T/dt ) + 1;
    tList = [ [i*dt for i in range( Nt )] ];

    # generate training data
    N0 = 10;
    X0 = np.vstack( (np.random.rand(Nx-2,N0), np.zeros( (Nx-1,N0) )) );
    xData, uData = generate_data( tList, model, X0, control=control, Nu=Nu );

    # formatting training data from xData and uData
    uStack = stack_data( uData, N0, Nu, Nt-1 );
    xStack = stack_data( xData[:,:-1], N0, Nx, Nt-1 );
    yStack = stack_data( xData[:,1:], N0, Nx, Nt-1 );

    XU0 = np.vstack( (X0, np.zeros( (Nu,N0) )) );
    X = np.vstack( (xStack, np.zeros( (Nu,N0*(Nt-1)) )) );
    Y = np.vstack( (yStack, uStack) );

    # learn Koopman operator
    kvar = KoopmanOperator( obs );
    kvar.edmd( X, Y, X0=XU0 );

    print( kvar );

    # simulate model
    kModel = lambda Psi, u: kvar.K@Psi;
    _, xList = simulateModelWithControl( obs( xu0 ), kModel,
        N=1000, sim=0 );

    # plot static results
    fig, axs = plt.subplots();
    axs.plot( xList[:Nx-1,:].T );
    plt.show();