function [Psi_xuh, meta] = obs(x)
    Psi_x = obs_x(x);
    Psi_u = obs_u(x);
    Psi_h = obs_h([x;0]);

    Psi_xuh = [Psi_x; kron(Psi_u, Psi_h)];

    meta.Nk = length(Psi_xuh);
end