function [Psi, meta] = obs(x)
    PsiX = obsX(x);
    PsiU = obsU(x);
    PsiH = obsH([x;0;0]);

    Psi = [PsiX; vec(kron(PsiH', PsiU))];

    meta.Nk = length(Psi);
end