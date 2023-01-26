function [Psi, meta] = obs(x)
    [PsiX, metaX] = obsX(x);
    [PsiU, metaU] = obsU(x);
    [h, metaH] = obsH([x;0;0]);

    Psi = [PsiX; kron(PsiU, h)];

    meta.X = 1:metaX.Nk;

    offset = metaX.Nk;
    for i = 1:metaH.Nk
        meta.("U"+i+"h") = (1:metaU.Nk) + offset;
        offset = offset + metaU.Nk;
    end

    meta.Nk = length(Psi);
end