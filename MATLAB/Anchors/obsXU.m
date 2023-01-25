function [PsiXU, metaXU] = obsXU(x)  
    [PsiX, metaX] = obsX(x);
    [PsiU, metaU] = obsU(x);
    [PsiH, metaH] = obsH([x;0;0]);

    PsiXU = [PsiX; kron(PsiU, PsiH)];

    labels = fieldnames(metaU);

    metaXU = metaX;
    for i = 1:length(labels)
        metaXU.(labels{i}) = metaU.(labels{i}) + metaXU.Nk;
    end

    metaXU.Nk = metaX.Nk*(1 + metaH.Nk);
end