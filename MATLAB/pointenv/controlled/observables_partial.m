function [dPsix, dPsiu] = observables_partial(x, u)
    obs_x = @(x) observables(x, u);
    dPsix = naps.fdm(obs_x, x);

    obs_u = @(u) observables(x, u);
    dPsiu = naps.fdm(obs_u, u);
end