function [PsiX, meta] = obsX(x)
    PsiX = x;

    meta.x = 1:4;
%     meta.c = 5;
    meta.Nk = 4;
end