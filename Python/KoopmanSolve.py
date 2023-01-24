import numpy as np

def KoopmanSolve(observables, X, Y, x0, U=None, meta=None, eps=None):
    # grab meta variable
    if meta is None:
        (_, meta) = observables(x0[:,0]);
    Nk = meta.Nk;

    TOL = 1e-12;

    # evaluate for observable functions over X and Y
    N0 = len(x0[0,:]);
    Mx = round(len(X[0,:])/N0);

    PsiX = np.zeros(Nk, N0*(Mx-1));
    PsiY = np.zeros(Nk, N0*(Mx-1));

    i = 0;
    j = 0;

    for n in range(N0):

        for m in range(Mx-1):

            i = i + 1;
            j = j + 1;

            if U is None:
                PsiX[:,j] = observables(X[:,i]);
                PsiY[:,j] = observables(Y[:,i]);
            else:
                PsiX[:,j] = observables(X[:,i], U[:,i]);
                PsiY[:,j] = observables(Y[:,i], U[:,i+1]);

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
