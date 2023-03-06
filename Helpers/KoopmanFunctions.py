import numpy as np

def vec(A):
    (n, m) = A.shape;
    return A.reshape(n*m,1);

class KoopmanOperator:
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

    def liftData(self, X, X0, obs=None):
        # default observation is obsX
        if obs is None:
            obs = self.obsX;

        # dimension variables
        Nk = obs()['Nk'];
        Mx = len(X[0]);
        Nx = len(X0);
        
        if len(X0.shape) > 1:
            N0 = len(X0[0]);
        else:
            N0 = 1;
        
        Nt = round(Mx/N0);

        dim = (N0, Nt, Nx, Nk);  # for return

        # observation initialization
        Psi = np.empty( (Nk, N0*(Nt-1)) );

        # loop variables
        i = 0;
        j = 0;

        for n in range(N0):

            for m in range(Nt-1):

                Psi_new = obs(X[:,i].reshape(Nx,1));
                Psi[:,j] = Psi_new.reshape(Nk,);

                i += 1;
                j += 1;

            i += 1;
            j = n*(Nt - 1);

        return Psi, dim;

    def edmd(self, X, Y, X0, eps=None):
        # tolerance variable
        TOL = 1e-12;

        # evaluate for observable functions over X and Y
        PsiX, (N0, Nt, Nx, Nk) = self.liftData(X, X0);
        PsiY, _ = self.liftData(Y, X0, self.obsY);

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

        # calculate residual error
        err = 0;
        for n in range(N0*(Nt-1)):
            err += np.linalg.norm(PsiY[:,n] - K@PsiX[:,n]);

        self.K = K;
        self.err = err;
        self.ind = ind;

        return K;

    def cd(self, X, Y, X0, G, eps=None):
        pass;
