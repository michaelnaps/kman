function [Psi,meta] = obs(X)
    x = X(1:4);  u = X(5:6);

    PsiX = obsX(x);
    PsiU = obsU(x);
    PsiH = u;

    Psi = [PsiX; kron(PsiU, PsiH)];
    meta.Nk = 6;
end