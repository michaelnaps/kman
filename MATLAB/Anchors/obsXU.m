function [PsiXU, metaXU] = obsXU(x, u)  
    [PsiX, metaX] = obsX(x);
    [PsiU, metaU] = obsU(x);

    PsiXU = [PsiX; PsiU*u];

    labels = fieldnames(metaU);

    metaXU = metaX;
    for i = 1:length(labels)
        metaXU.(labels{i}) = metaU.(labels{i}) + metaXU.Nk;
    end
end