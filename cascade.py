# script for model equation testing
from diffdrive import *

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

    # initialize states
    x0 = [0,0,pi];
    xd = [0,0,0];
    uinit = [0 for i in range(Nu*PH)];

    # simulation variables and data gen.
    T = 10;  Nt = round(T/dt)+1;
    tList = [[i*dt for i in range(Nt)]];

    # create MPC class variable
    dt_mpc = 0.01;
    model_type = 'discrete';
    max_iter = 25;
    params = Parameters(x0, xd, buffer_length=25);
    mpc_var = mpc.ModelPredictiveControl('ngd', model, cost, params, Nu,
        num_ssvar=Nx, PH_length=PH, knot_length=kl, time_step=dt_mpc,
        max_iter=max_iter, model_type=model_type);
    mpc_var.setAlpha(0.01);

    # generate initial conditions for training
    A = 10;
    N0 = 10;
    X0 = 2*A*np.random.rand(Nx,N0) - A;
    X, Y, XU0 = createDynamicSets(tList, X0);

    kxvar = kman.KoopmanOperator(obsXUH, obsXU, mpc_var);
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
