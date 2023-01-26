function [PsiU, metaU] = obsU(x)
    % here, for simplicity: Psi_u = Psi_x
    [PsiX, metaX] = obsX(x);

    PsiU = [PsiX; 1];

    labels = fieldnames(metaX);
    for i = 1:length(labels)
        if labels{i} ~= "Nk"
            metaU.(labels{i}+"u") = metaX.(labels{i});
        end
    end

    metaU.uc = metaX.Nk + 1;

    % meta data vaiable
    metaU.Nk = metaX.Nk + 1;
end