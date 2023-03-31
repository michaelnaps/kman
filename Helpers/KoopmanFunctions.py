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
    Mx = len(X[0]);
    Nx = len(X0);

    if len(X0.shape) > 1:
        N0 = len(X0[0]);
    else:
        N0 = 1;

    Nt = round(Mx/N0);

    return N0, Nt, Nx, Nk;

# iterative least squares algorithm
def iterative_lstsq(klist, flist, X, Y, X0, TOL=1e-3):
    # dimension variables and vectorized klist
    N = len(klist);
    (N0, Nt, _, _) = dimnData(X, X0);
    Nklist = [kvar.K.shape for kvar in klist];
    kvlist = [vec( kvar.K ) for kvar in klist];

    PsiX = [None for i in range(N)];
    PsiY = [None for i in range(N)];
    for i, kvar in enumerate(klist):
        PsiX[i], _ = kvar.liftData(X, X0);
        PsiY[i], _ = kvar.liftData(Y, X0, kvar.obsY);

    # primary algorithm loop
    dK = 1;  count = 0;
    while dK > TOL:
        dK = 0;
        for i, f in enumerate(flist):
            kcopy = kvlist[i];

            PsiShiftX = f(klist, PsiX[i]);

            print('______');
            print(PsiX[i].shape, PsiShiftX.shape, PsiY[i].shape);

            G = PsiShiftX@PsiShiftX.T;
            A = PsiShiftX@PsiY[i].T;
            print(M.shape, G.shape, A.shape);

            kvlist[i] = np.linalg.lstsq(G, A);
            klist[i].K = nvec(kvlist[i], Nklist[i][0], Nklist[i][1]);

            dK += np.linalg.norm( kvlist[i] - kcopy );
        count += 1
        print(count, ': %.5e' % dK);

    return klist;

# cascade extended dynamic mode decomposition algorithm
def cascade_edmd(klist, flist, X, Y, X0, TOL=1e-3):
    # calculation loop for BCD
    dK = 1;  count = 0;
    while dK > TOL:
        dK = 0;
        for i, f in enumerate(flist):
            kcopy = klist[i].K;

            if f is not None:
                M = f(klist[:i]);
                klist[i].setShiftMatrix(M);

            klist[i].edmd(X, Y, X0);

            dK += np.linalg.norm( klist[i].K - kcopy );
        count += 1
        print(count, ': %.5e' % dK);

    return klist;

# Koopman Operator class description
class KoopmanOperator:
    # initialize class
    def __init__(self, obsX, obsY=None, params=None, M=None, K=None):
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

        if M is None:
            self.M = np.eye( self.metaX['Nk'] );
        else:
            self.M = M;

        if K is None:
            self.K = np.eye( self.metaY['Nk'], self.M.shape[0] );
        else:
            self.K = K;

        self.eps = None;
        self.params = params;

        # accuracy variables
        self.err = -1;
        self.ind = None;

    # when asked to print - return operator
    def __str__(self):
        line1 = 'Error: %.3e' % self.err;
        line2 = ', Shape: (' + str(self.K.shape[0]) + ', ' + str(self.K.shape[1]) + ')\n';
        line3 = np.array2string( self.K, precision=2, suppress_small=1 );
        return line1 + line2 + line3;

    # set the shift matrix after init
    def setShiftMatrix(self, M):
        self.M = M;
        return self;

    # lift data from state space to function domain
    def liftData(self, X, X0, obs=None):
        # default observation is obsX
        if obs is None:
            M = self.M;
            obs = self.obsX;
        else:
            M = np.eye( obs()['Nk'] )
        (N0, Nt, Nx, Nk) = dimnData(X, X0, obs);

        # observation initialization
        NkM = M.shape[0];
        Psi = np.empty( (NkM, N0*Nt) );

        for n in range(N0*Nt):
            Psi_new = M@obs(X[:,n,None]);
            Psi[:,n] = Psi_new.reshape(NkM,);

        return Psi, NkM;

    # residual error over supplied data set
    def resError(self, X, Y, X0, K=None):
        # set operator
        if K is None:
            K = self.K;

        # get data parameters
        (N0, Nt, Nx, _) = dimnData(X, X0);
        PsiX, NkX = self.liftData(X, X0);
        PsiY, NkY = self.liftData(Y, X0, self.obsY);

        # calculate residual error
        err = 0;
        for n in range(N0*(Nt-1)):
            err += np.linalg.norm(PsiY[:,n,None] - K@PsiX[:,n,None]);

        self.err = err;
        return err;

    # extended dynamical mode decomposition (EDMD)
    def edmd(self, X, Y, X0, G=None, A=None, eps=None):
        # tolerance variable
        TOL = 1e-12;

        # evaluate for observable functions over X and Y
        (N0, Nt, Nx, _) = dimnData(X, X0);

        if G is None and A is None:
            PsiX, NkX = self.liftData(X, X0);
            PsiY, NkY = self.liftData(Y, X0, self.obsY);

        # perform EDMD
        # create matrices for least squares
        #   K = inv(G)*A
        # (according to abraham, model-based)
        if G is None:
            G = 1/(N0*Nt) * (PsiX @ PsiX.T);
        if A is None:
            A = 1/(N0*Nt) * (PsiX @ PsiY.T);

        (U, S, V) = np.linalg.svd(G);

        if eps is None:
            eps = TOL*max(S);

        ind = S > eps;

        U = U[:,ind];
        V = V[ind,:].T;

        S = S[ind];
        Sinv = np.diag([1/S[i] for i in range(len(S))]);

        # solve for the Koopman operator
        K = A.T @ (U @ Sinv @ V.T);

        self.K = K;
        self.resError(X, Y, X0);
        self.ind = ind;

        return K;
