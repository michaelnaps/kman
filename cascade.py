# script for model equation testing
from diffdrive import *

def controlDynamics(u, x):
    alpha = 0.01;
    g = mpc_var.gradient(x, u);
    un = u - alpha*g;
    return un;

def createDynamicSets(tList, X0):
    # model function for training syntax
    modelTrain = lambda x, u: np.array( model(x,u,None) ).reshape(Nx,1);
    controlTrain = lambda x: np.random.rand(Nu,1);
    # np.array( mpc_var.solve(x, uinit)[0][:Nu] ).reshape(Nu,1);
    xTrain, uTrain = data.generate_data(tList, modelTrain, X0, controlTrain, Nu);

    # split training data into X and Y sets
    uStack = data.stack_data(uTrain, N0, Nu, Nt-1);
    xStack = data.stack_data(xTrain[:,:-1], N0, Nx, Nt-1);
    yStack = data.stack_data(xTrain[:,1:], N0, Nx, Nt-1);

    X = np.vstack( (xStack, uStack) );
    Y = np.vstack( (yStack, uStack) );

    # reshape initial condition set
    XU0 = np.vstack( (X0, np.zeros( (Nu, N0) )) );

    return X, Y, XU0;

if __name__ == "__main__":
    # observable dimensions variables
    NkX = obsX()['Nk'];  # for reference
    NkU = obsU()['Nk'];
    NkH = obsH()['Nk'];

    # simulation variables and data gen.
    T = 10;  Nt = round(T/dt)+1;
    tList = [[i*dt for i in range(Nt)]];

    # generate initial conditions for training
    A = 10;
    N0 = 10;
    X0 = 2*A*np.random.rand(Nx,N0) - A;
    X, Y, XU0 = createDynamicSets(tList, X0);

    kxvar = kman.KoopmanOperator(obsXUH, obsXU);
    Kx = kxvar.edmd(X, Y, XU0);

    print('Kx:\n', kxvar);
    print('Kx.PsiX:\n', kxvar.K[:NkX,:].T);
    print('Kx.PsiU:\n', kxvar.K[NkX:,:].T);

    # evaluate the behavior of Kx with remeasurement function
    x0ref = np.array( x0 )[:,None];
    uref = np.array( [[1],[2]] );
    xTest, PsiTest = posTrackingNoControl(tList, kxvar, x0ref, uref);

    # plot test results
    plotcomp(tList, xTest, PsiTest);
