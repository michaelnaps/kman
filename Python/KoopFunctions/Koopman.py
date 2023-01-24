import numpy as np

def Koopman(observables, X, Y, x0, uList, meta=None, eps=None):

    if meta is None:
        (_, meta) = observables(x0[:,0], uData[:,1]);
    Nk = meta.Nk;

    TOL = 1e-12;

    # evaluate for observable functions over X and Y
    N0 = len(x0[0,:]);
    Mx = round(len(X[0,:])/N0);

    PsiX = np.nan(Nk, N0*(Mx-1));
    PsiY = np.nan(Nk, N0*(Mx-1));

    # CONTINUE FROM LINE 24 IN MATLAB VERSION

    K = 1;
    acc = 1;
    ind = 1;
    err = 1;

    return (K, acc, ind, err);
