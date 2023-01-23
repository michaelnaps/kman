function [Psi_x, meta] = obsX(x)
    Psi_x = [x; 1];

    meta.x = 1:4;
    meta.c = 5;
    meta.Nk = 5;
end