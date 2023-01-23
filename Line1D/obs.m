function [Psi_xuh, meta] = obs(x)
    Psi_x = obs_x(x);
    Psi_u = obs_u(x);
    Psi_h = obs_h([x;0]);

    Psi_xuh = [Psi_x; vec(kron(Psi_h', Psi_u))];

    meta.Nk = length(Psi_xuh);
end