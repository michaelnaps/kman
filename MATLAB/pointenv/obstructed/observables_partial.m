function [dPsix, dPsiu] = observables_partial(x, u, obsFun)
    obs_x = @(x) obsFun(x, u);
    dPsix = naps.fdm(obs_x, x);

    obs_u = @(u) obsFun(x, u);
    dPsiu = naps.fdm(obs_u, u);
end