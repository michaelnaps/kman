import numpy as np

class KoopmanOperator:
    def __init__(self, obsX, obsY=None):
        # function parameters
        self.obsX = obsX;
        if obsY is None:
            self.obsY = obsX;
        else:
            self.obsY = obsY;


        # Koopman parameters
        self.K = None;
        self.metaX = self.obsX();
        self.metaY = self.obsY();
        self.eps = None;

        # accuracy variables
        self.err = None;
        self.ind = None;

    def edmd(self, X, Y, X0, eps=None):
        # tolerance variable
        TOL = 1e-12;

        # get class variables
        obsX = self.obsX;
        obsY = self.obsY;
        NkX = self.metaX['Nk'];
        NkY = self.metaY['Nk'];

        # evaluate for observable functions over X and Y
        Mx = len(X[0]);
        Nx = len(X0);
        if len(X0.shape) > 1:
            N0 = len(X0[0]);
        else:
            N0 = 1;
        Nt = round(Mx/N0);

        PsiX = np.empty( (NkX, N0*(Nt-1)) );
        PsiY = np.empty( (NkY, N0*(Nt-1)) );

        i = 0;
        j = 0;

        for n in range(N0):

            for m in range(Nt-1):

                PsiX_new = obsX(X[:,i].reshape(Nx,1));
                PsiY_new = obsY(Y[:,i].reshape(Nx,1));

                PsiX[:,j] = PsiX_new.reshape(NkX,);
                PsiY[:,j] = PsiY_new.reshape(NkY,);

                i += 1;
                j += 1;

            i += 1;
            j = n*(Nt - 1);

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
        # K = (V @ (1/S) @ U.T) @ A;
        K = (V @ Sinv @ U.T) @ A;
        K = K.T;

        # calculate residual error
        err = 0;
        for n in range(N0*(Nt-1)):
            err += np.linalg.norm(PsiY[:,n] - K@PsiX[:,n]);

        self.K = K;
        self.err = err;
        self.ind = ind;

        return K;
