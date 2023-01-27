function [Psi_u, meta] = obs_u(x)
    % here, for simplicity: Psi_u = Psi_x
    Psi_u = 1;

    % meta data variable
%     meta.xu = [1,2];
%     meta.xxu = [3,4,5,6];
    meta.cu = 1;
    meta.Nk = 1;
end