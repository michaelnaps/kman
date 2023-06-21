import numpy as np

# matrix -> vector form
def vec(A):
    (n, m) = A.shape;
    return A.reshape(n*m,1);

# vector -> matrix form
def nvec(a, n=None, m=None):
    if m is None:
        m = n;

    if n is None:
        N = len( a );
        n = round( np.sqrt(N) );
        m = round( n );

    return a.reshape(n,m);

# return the dimensions of a data set
def dimnData(X, X0, obs=None):
    # default observation is obsX
    if obs is None:
        Nk = None;
    else:
        Nk = obs()['Nk'];

    # dimension variables
    Tx = len(X[0]);
    Nx = len(X0);

    if len(X0.shape) > 1:
        N0 = len(X0[0]);
    else:
        N0 = 1;

    Nt = round(Tx/N0);

    return N0, Nt, Nx, Nk;

# cascade extended dynamic mode decomposition algorithm (recursive)
def cascade_edmd(klist, flist, X, Y, X0):
    # if number of operators is 1, solve edmd
    if len(klist) == 1:
        knvar = klist[0].edmd(X, Y, X0);
        return knvar;

    # when N(klist) != 1, cut front operator can reenter function
    K = cascade_edmd(klist[1:], flist[1:], X, Y, X0);

    # give the resulting operator list to the shift function and solve
    klist[0].setShiftMatrix( flist[0](K) );
    klist[0].edmd(X, Y, X0);

    return klist;

# Koopman Operator class description
class KoopmanOperator:
    # initialize class
    def __init__(self, obsX, obsY=None, params=None, T=None, K=None):
        # function parameters
        if params is None:
            self.obsX = obsX;
            if obsY is None:
                self.obsY = self.obsX;
            else:
                self.obsY = obsY;
        else:
            self.obsX = lambda x=None: obsX(x, params);
            if obsY is None:
                self.obsY = self.obsX;
            else:
                self.obsY = lambda y=None: obsY(y, params);

        # Koopman parameters
        self.metaX = self.obsX();
        self.metaY = self.obsY();

        if T is None:
            self.T = np.eye( self.metaX['Nk'] );
        else:
            self.T = T;

        if K is None:
            self.K = np.eye( self.metaY['Nk'], self.T.shape[0] );
        else:
            self.K = K;

        self.eps = None;
        self.params = params;

        # accuracy variables
        self.err = -1;
        self.ind = None;

    # when asked to print - return operator
    def __str__(self):
        line1 = 'Error: %.5e' % self.err;
        line2 = ', Shape: (' + str(self.K.shape[0]) + ', ' + str(self.K.shape[1]) + ')\n';
        line3 = np.array2string( self.K, precision=5, suppress_small=1 );
        return line1 + line2 + line3;

    # set the shift matrix after init
    def setShiftMatrix(self, T):
        self.T = T;
        return self;

    # lift data from state space to observation space
    def liftData(self, X, X0, obs=None):
        # default observation is obsX
        if obs is None:
            T = self.T;
            obs = self.obsX;
        else:
            T = np.eye( obs()['Nk'] )
        (N0, Nt, Nx, Nk) = dimnData(X, X0, obs);

        # observation initialization
        NkT = T.shape[0];
        Psi = np.empty( (NkT, N0*Nt) );

        for n in range(N0*Nt):
            Psi_new = T@obs(X[:,n,None]);
            Psi[:,n] = Psi_new.reshape(NkT,);

        return Psi, NkT;

    # residual error over data set (pre-lift)
    def resErrorXY(self, X, Y, X0, K=None):
        # set operator
        if K is None:
            K = self.K;

        # lift data into observation space
        (N0, Nt, Nx, _) = dimnData(X, X0);
        PsiX, NkX = self.liftData(X, X0);
        PsiY, NkY = self.liftData(Y, X0, self.obsY);

        # calculate residual error
        err = 0;
        for n in range(N0*(Nt-1)):
            err += np.linalg.norm(PsiY[:,n,None] - K@PsiX[:,n,None])**2;

        self.err = err;
        return err;

    # residual error over data set (post-lift)
    def resErrorLifted(self, PsiX, PsiY, PsiX0, K=None):
        # set operator
        if K is None:
            K = self.K;

        # data dimensions
        (N0, Nt, Nx, _) = dimnData(PsiX, PsiX0);

        # calculate residual error
        err = 0;
        for n in range(N0*(Nt-1)):
            err += np.linalg.norm(PsiY[:,n,None] - K@PsiX[:,n,None])**2;

        self.err = err;
        return err;

    # dynamic mode decomposition (DMD)
    def dmd(self, X, Y, X0, eps=None):
        # get data dimensions (optional step)
        TOL = 1e-12;
        (N0, Nt, Nx, _) = dimnData(X, X0);

        # perform DMD on given data
        # create matrices for least squares
        #   K = inv(G)*A
        # (according to abraham, model-based)
        G = 1/(N0*Nt) * (X @ X.T);
        A = 1/(N0*Nt) * (X @ Y.T);

        # get SVD matrices
        (U, S, V) = np.linalg.svd(G);

        # get priority functions
        if eps is None:
            eps = TOL*max(S);
        ind = S > eps;

        # truncate space for prioritized functions
        U = U[:,ind];
        S = S[ind];
        V = V[ind,:].T;

        # invert S values and create matrix
        Sinv = np.diag([1/S[i] for i in range(len(S))]);

        # solve for the Koopman operator
        K = A.T @ (U @ Sinv @ V.T);

        # update class variables
        self.K = K;
        self.resErrorLifted(X, Y, X0);
        self.ind = ind;

        return self;

    # extended dynamic mode decomposition (EDMD)
    def edmd(self, X, Y, X0, eps=None):
        # lift data to observation space
        PsiX, _ = self.liftData(X, X0);
        PsiY, _ = self.liftData(Y, X0, self.obsY);

        # give lifted data to DMD algorithm
        return self.dmd(PsiX, PsiY, X0, eps=eps);