function [u] = ModelQP(xG, x0, world)

    Nw = length(world);

    pt = x0;
    obs_dist = NaN(Nw, 1);
    obs_grad = NaN(Nw, 2);
    for i = 1:Nw
        obs_dist(i) = distance(world(i), pt);
        obs_grad(i,:) = distance_grad(world(i), pt, obs_dist(i));
    end

    u_ref = (pt- xG) / norm(pt - xG);

    u = qp_supervisor(-obs_grad, -obs_dist', -u_ref')';

end