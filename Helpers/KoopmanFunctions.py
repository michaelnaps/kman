import numpy as np

def vec(A):
    (n, m) = A.shape;
    return A.reshape(n*m,1);

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

# block coordinate descent (CD)
def bcd(Klist, flist, X, Y, X0, TOL=1e-3):
    # operator dimensions
    N = len(Klist);
    (N0, Nt, _, _) = dimnData(X, X0);

    # lift data into the appropriate function spaces
    Glist = [None for i in range(N)];
    Alist = Glist;
    for i, kvar in enumerate(Klist):
        PsiX, _ = kvar.liftData(X, X0);
        PsiY, _ = kvar.liftData(Y, X0, kvar.obsY);
        Glist[i] = 1/(N0*(Nt - 1)) * np.sum(PsiX, axis=1)[:,None];
        Alist[i] = 1/(N0*(Nt - 1)) * np.sum(PsiY, axis=1)[:,None];

    # error loop for BCD
    dK = 1;  count = 0;
    Kcopy = [Klist[i] for i in range(N)];
    while np.linalg.norm(dK) > TOL:
        dK = 0;
        for i, f in enumerate(flist):
            NkX = Klist[i].metaX['Nk'];
            NkY = Klist[i].metaY['Nk'];

            M = f(Klist, Glist[i]);
            Ksoln = np.linalg.lstsq(M, Alist[i], rcond=None);
            Klist[i].K = nvec( Ksoln[0], NkX, NkY );

            dK += np.linalg.norm(Klist[i].K - Kcopy[i].K);
            Kcopy[i] = Klist[i]; 
        count += 1
        print(count);

    # calculate the resulting error for each operator
    for i in range(N):
        Klist[i].err = Klist[i].resError(X, Y, X0);

    Kl = Klist;
    return Kl;

# Koopman Operator class description
class KoopmanOperator:
    # initialize class
    def __init__(self, obsX, obsY=None, params=None):
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
        self.K = np.eye(self.metaX['Nk'], self.metaY['Nk']);
        self.eps = None;
        self.params = params;

        # accuracy variables
        self.err = -1;
        self.ind = None;

    # when asked to print - return operator
    def __str__(self):
        return "Error: %.3f\n" % self.err + np.array2string( self.K, precision=2, suppress_small=1 );

    # lift data from state space to function domain
    def liftData(self, X, X0, obs=None):
        # default observation is obsX
        if obs is None:
            obs = self.obsX;
        (N0, Nt, Nx, Nk) = dimnData(X, X0, obs);

        # observation initialization
        Psi = np.empty( (Nk, N0*(Nt-1)) );

        # loop variables
        i = 0;
        j = 0;

        for n in range(N0):

            for m in range(Nt-1):

                Psi_new = obs(X[:,i,None]);
                Psi[:,j] = Psi_new.reshape(Nk,);

                i += 1;
                j += 1;

            i += 1;
            j = n*(Nt - 1);

        return Psi, Nk;

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
            err += np.linalg.norm(PsiY[:,n] - K@PsiX[:,n]);

        return err;

    # extended dynamical mode decomposition (EDMD)
    def edmd(self, X, Y, X0, eps=None):
        # tolerance variable
        TOL = 1e-12;

        # evaluate for observable functions over X and Y
        (N0, Nt, Nx, _) = dimnData(X, X0);
        PsiX, NkX = self.liftData(X, X0);
        PsiY, NkY = self.liftData(Y, X0, self.obsY);

        # perform EDMD
        # create matrices for least squares
        #   K = inv(G)*A
        # (according to abraham, model-based)
        G = 1/(N0*(Nt - 1)) * (PsiX @ PsiX.T);
        A = 1/(N0*(Nt - 1)) * (PsiX @ PsiY.T);

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
        self.err = self.resError(X, Y, X0, self.K);
        self.ind = ind;

        return K;



        
