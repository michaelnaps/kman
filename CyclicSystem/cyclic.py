import sys
sys.path.insert(0, '../');

from root import *

# observable identification: control
def obs(X=None):
    if X is None:
        meta = {'Nk':Nx+Nu+3};
        return meta;

    x = X[:Nx];
    u = X[Nx:];

    xTrig = np.array( [np.cos( x[2] ), np.sin( x[2] )] );
    Psi = np.vstack( (x, u, xTrig, [1]) );

    return Psi;

# Main execution block.
if __name__ == '__main__':
    x0 = np.array( [[7.5],[0],[0]] );
    xu0 = np.vstack( (x0, [[0],[0],[0]]) );

    # simulation time variables
    T = 10;  Nt = round( T/dt ) + 1;
    tList = [ [i*dt for i in range( Nt )] ];

    # generate training data
    xData, uData = generate_data(tList, model, x0,
        control=control, Nu=Nu)

    # format data for training
    X = np.vstack( (xData[:,:-1], uData) );
    Y = np.vstack( (xData[:,1:], uData) );

    # learn Koopman operator
    kvar = KoopmanOperator( obs );
    kvar.edmd( X, Y );

    print( kvar );

    # simulate model
    kModel = lambda Psi, u: kvar.K@Psi;
    simulateModelWithControl( obs( xu0 ), kModel, N=1000 );
    print("Animation finished...");