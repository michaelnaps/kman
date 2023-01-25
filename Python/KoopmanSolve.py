import numpy as np

def KoopmanSolve(observables, Nk, X, Y, x0, U=None, meta=None, eps=None):
    # tolerance variable
    TOL = 1e-12;

    # evaluate for observable functions over X and Y
    N  = len(x0);
    N0 = len(x0[0]);
    Mx = round(len(X[0])/N0);

    PsiX = np.zeros( (Nk, N0*(Mx-1)) );
    PsiY = np.zeros( (Nk, N0*(Mx-1)) );

    i = 0;
    j = 0;

    for n in range(N0):

        for m in range(Mx-1):

            i = i + 1;
            j = j + 1;

            if U is None:
                (PsiX_new, _) = observables(X[:,i].reshape(N,1));
                (PsiY_new, _) = observables(Y[:,i].reshape(N,1));

                PsiX[:,j] = PsiX_new.reshape(Nk,);
                PsiY[:,j] = PsiY_new.reshape(Nk,);
            else:
                (PsiX[:,j], _) = observables(X[:,i], U[:,i]);
                (PsiY[:,j], _) = observables(Y[:,i], U[:,i+1]);

        i = i + 1;
        j = n*(Mx - 1);


    # perform EDMD
    # create matrices for least squares
    #   K = inv(G)*A
    # (according to abraham, model-based)
    G = 1/(N0*(Mx - 1)) * (PsiX @ PsiX.T);
    A = 1/(N0*(Mx - 1)) * (PsiX @ PsiY.T);

    (U, S, V) = np.linalg.svd(G);

    if eps is None:
        eps = TOL*max(np.diag(S));

    ind = np.diag(S) > eps;

    U = U[:,ind];
    S = S[ind,ind];
    V = V[ind,:];

    # solve for the Koopman operator
    K = (V @ (1/S) @ U.T) @ A;
    K = K.T;

    # calculate residual error
    err = 0;
    for n in range(N0*(Mx-1)):
        err += np.linalg.norm(PsiY[:,n] - K@PsiX[:,n]);

    K = 1;
    acc = 1;
    ind = 1;
    err = 1;

    return (K, err, ind);
