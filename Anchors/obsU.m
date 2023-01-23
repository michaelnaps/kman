function [PsiU, meta] = obsU(x)
    % here, for simplicity: Psi_u = Psi_x
    PsiX = obsX(x);

    PsiU = [eye(2); PsiX, PsiX];

    % meta data variable
    meta.u = 1:2;
    meta.xu = 3:6;
    meta.cu = 7;
    meta.Nk = 7;
end