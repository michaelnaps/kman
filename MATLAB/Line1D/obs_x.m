function [Psi_x, meta] = obs_x(x)
    Nx = length(x);
    Nxx = Nx*Nx;

    Psi_x = x;

    % meta data variable
    meta.x = [1,2];
    meta.Nk = 2;
end