function [Psi, meta] = obs(x)
    PsiX = obsX(x);
    PsiU = obsU(x);
    h = obsH([x;0;0]);

    Psi = [PsiX; kron(PsiU, h)];

    meta.Nk = length(Psi);
end