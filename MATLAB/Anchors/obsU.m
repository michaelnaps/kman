function [PsiU, meta] = obsU(x)
    % here, for simplicity: Psi_u = Psi_x
    [PsiX, metaX] = obsX(x);

    PsiU = PsiX;

    % meta data vaiable
    meta.Nk = metaX.Nk;
end