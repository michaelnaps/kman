import numpy as np

class KoopmanOperator:
    def __init__(self, observables):
        # function parameters
        self.obs = observables;

        # Koopman parameters
        self.K = None;
        self.Nk = self.obs();
        self.eps = None;

        # accuracy variables
        self.err = None;
        self.ind = None;

    def edmd(self, X, Y, x0, eps=None):
        # tolerance variable
        TOL = 1e-12;

        # get class variables
        observables = self.obs;
        Nk = self.Nk;

        # evaluate for observable functions over X and Y
        Nx = len(x0);
        N0 = len(x0[0]);
        Mx = round(len(X[0])/N0);

        PsiX = np.ones( (Nk, N0*(Mx-1)) );
        PsiY = np.zeros( (Nk, N0*(Mx-1)) );

        i = 0;
        j = 0;

        for n in range(N0):

            for m in range(Mx-1):

                PsiX_new = observables(X[:,i].reshape(Nx,1));
                PsiY_new = observables(Y[:,i].reshape(Nx,1));

                PsiX[:,j] = PsiX_new.reshape(Nk,);
                PsiY[:,j] = PsiY_new.reshape(Nk,);

                i += 1;
                j += 1;

            i += 1;
            j = n*(Mx - 1);


        # perform EDMD
        # create matrices for least squares
        #   K = inv(G)*A
        # (according to abraham, model-based)
        G = 1/(N0*(Mx - 1)) * (PsiX @ PsiX.T);
        A = 1/(N0*(Mx - 1)) * (PsiX @ PsiY.T);

        (U, S, V) = np.linalg.svd(G);

        if eps is None:
            eps = TOL*max(S);

        ind = S > eps;

        U = U[:,ind];
        V = V[:,ind];

        S = S[ind];
        Sinv = np.diag([1/S[i] for i in range(len(S))]);

        # solve for the Koopman operator
        # K = (V @ (1/S) @ U.T) @ A;
        K = (V @ Sinv @ U.T) @ A;
        K = K.T;

        # calculate residual error
        err = 0;
        for n in range(N0*(Mx-1)):
            err += np.linalg.norm(PsiY[:,n] - K@PsiX[:,n]);

        self.K = K;
        self.err = err;
        self.ind = ind;

        return K;
