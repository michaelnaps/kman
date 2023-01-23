function [Psi_x, meta] = obsX(x)
    Psi_x = [x; 1];

    meta.x = [1,2];
    meta.c = 3;
    meta.Nk = 3;
end