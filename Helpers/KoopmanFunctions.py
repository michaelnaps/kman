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

def sind(A, s):
    (n, m) = A.shape;

    if s > n*m-1:
        print("ERROR: Index to large for matrix.");
        return None;
    
    i = 0;
    j = s;
    while j > m-1:
        i = i + 1;
        j = j - m;

    return i, j;

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
        self.K = None;
        self.metaX = self.obsX();
        self.metaY = self.obsY();
        self.eps = None;

        # accuracy variables
        self.err = None;
        self.ind = None;

    # return the dimensions of the data set
    def dimnData(self, X, X0, obs=None):
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

    # lift data from state space to function domain
    def liftData(self, X, X0, obs=None):
        # default observation is obsX
        if obs is None:
            obs = self.obsX;
        (N0, Nt, Nx, Nk) = self.dimnData(X, X0, obs);

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
        (N0, Nt, Nx, _) = self.dimnData(X, X0);
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
        (N0, Nt, Nx, _) = self.dimnData(X, X0);
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

    # finite difference method (FDM)
    def fdm(self, K, s, X, Y, X0, h=1e-3):
        ij = sind(K, s);

        Kn = K.copy();  Kp = K.copy();
        Kn[ij] = Kp[ij] - h;
        Kp[ij] = Kp[ij] + h;

        ep = self.resError(X, Y, X0, Kp);
        en = self.resError(X, Y, X0, Kn);

        return (ep - en)/(2*h);

    # block coordinate descent (CD)
    def bcd(self, Ml, X, Y, X0, Kl=None, eps=1e-3):
        # evaluate for observable functions over X and Y
        (N0, Nt, Nx, Nk) = self.dimnData(X, X0);
        PsiX, NkX = self.liftData(X, X0);
        PsiY, NkY = self.liftData(Y, X0, self.obsY);

        # initialize operator matrices
        if Kl is None:
            Kl = [vec( np.eye(Nk) ) for i in len(Ml)];

        # error loop for BCD
        dK = 1;
        while dK > eps:
            for i, k in enumerate(K):
                m = Ml[i](K, PsiX);
                K[i] = np.linalg.lstsq(PsiX, m);
                print(i);
                
        
        self.K = Kl;

        return Kl;

        
