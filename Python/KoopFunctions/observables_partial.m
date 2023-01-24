function [dPsix, dPsiu] = observables_partial(x, u, world)

    obs_x = @(x) observables(x, u, world);
    dPsix = fdm(obs_x, x);

    obs_u = @(u) observables(x, u, world);
    dPsiu = fdm(obs_u, u);

end