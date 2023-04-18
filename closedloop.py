from anchors import *

# closed-loop observation functions
def obsXU(X=None):
    if X is None:
        meta = {'Nk': 3*Nx+2*Nu+Na+1};
        return meta;

    x = X[:Nx];
    u = X[Nx:];

    xx = np.multiply(x,x);
    uu = np.multiply(u,u);
    xu = np.multiply(x,u);

    d = np.empty( (Na,1) );
    # xa = np.empty( (Nx,Na) );
    # ua = np.empty( (Nu,Na) );

    for i, a in enumerate(aList.T):
        d[i] = (x - a[:,None]).T@(x - a[:,None]);
        # xa[:,i] = np.multiply(x,a[:,None])[:,0];
        # ua[:,i] = np.multiply(u,a[:,None])[:,0];

    Psi = np.vstack( (x, d, xx, u, uu, xu, 1) );

    return Psi;

def obsX(X=None):
    if X is None:
        meta = {'Nk': 2*Nx+Na+Nx*Na};
        return meta;

    x = X[:Nx];

    xx = np.multiply(x,x);

    d = np.empty( (Na,1) );
    xa = np.empty( (Nx,Na) );

    for i, a in enumerate(aList.T):
        d[i] = (x - a[:,None]).T@(x - a[:,None]);
        xa[:,i] = np.multiply(x,a[:,None])[:,0];

    Psi = np.vstack( (x, d, xx, vec(xa)) );

    return Psi;

# main execution block
if __name__ == '__main__':
    # simulation data (for training)
    T = 100;  Nt = round(T/dt)+1;
    tList = [[i*dt for i in range(Nt)]];

    # generate data
    N0 = 1;
    X0 = 10*np.random.rand(Nx,N0) - 5;
    randControl = lambda x: 5*np.random.rand(Nu,1)-2.5;
    xData, uRand = data.generate_data(tList, model, X0,
        control=randControl, Nu=Nu);

    # stack data appropriately
    uStack = data.stack_data(uRand, N0, Nu, Nt-1);
    xStack = data.stack_data(xData[:,:-1], N0, Nx, Nt-1);
    yStack = data.stack_data(xData[:,1:], N0, Nx, Nt-1);

    # create data tuples for training
    XU0 = np.vstack( (X0, np.zeros( (Nu,N0) )) );
    X = np.vstack( (xStack, uStack) );
    Y = np.vstack( (yStack, uStack) );

    # initialize operator
    kvar = KoopmanOperator( obsXU, obsX );
    print( kvar.edmd(X, Y, XU0) );

    # # demonstrate closed-loop results
    # x0 = np.random.rand(Nx,1);
    # xu0 = np.vstack( (x0, np.zeros( (Nu,1) )) );
    # Psi0 = obsX(xu0);

    # # oscillation in shape of circle?
    # NkX = obsX()['Nk'];
    # def rmes(PsiX):
    #     PsiX = PsiX.reshape(NkX,1);
    #     x = PsiX[:Nx];

    #     u = randControl(x);
    #     print(u);

    #     uu = np.multiply(u,u);
    #     xu = np.multiply(x,u);
    #     ua = np.empty( (Nu,Na) );

    #     for i, a in enumerate(aList.T):
    #         ua[:,i] = np.multiply(u,a[:,None])[:,0];

    #     Psi = np.vstack( (PsiX, u, uu, xu, vec(ua)) );

    #     return kvar.K@Psi;

    # PsiData = data.generate_data(tList, rmes, Psi0)[0];
    # xData = PsiData[:Nx,:];

    # fig, axs = plt.subplots();
    # axs.plot(xData[:,0], xData[:,1]);
    # plt.show();