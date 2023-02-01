function [Psi_h, meta] = obs_h(X)
    % h(x) is composed of state and measured data
    x = X(1:2);
    u = X(3);

    Psi_h = [x; u];

    meta.xh = [1,2];
    meta.uh = 3;
%     meta.c = 4;
    meta.Nk = 3;
end