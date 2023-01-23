function [PsiU, meta] = obsU(x)
    % here, for simplicity: Psi_u = Psi_x
    PsiX = obsX(x);

    PsiU = [PsiX, PsiX];

    % meta data variable
    meta.xu = [1,2];
    meta.cu = 3;
    meta.Nk = 3;
end