function [Psi_x, meta] = obs_x(x)
    Nx = length(x);
    Nxx = Nx*Nx;

    Psi_x = [x; 1];

    % meta data variable
    meta.x = [1,2];
%     meta.xx = [3,4,5,6];
    meta.c = 3;
    meta.Nk = 3;
end