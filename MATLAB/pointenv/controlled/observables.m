function [psi] = observables(q, Q, env)

    if nargin < 3
        env = struct;
        env.xRange = [0, 1];
        env.yRange = [0, 1];
        env.maxVel = 1;
    end

    if nargin < 2
        Q = 3;
    end

    Ns = length(q);
    Nk = Ns*Q;

    x  = q(1);  dx = q(3);
    y  = q(2);  dy = q(4);

    lx = max(abs(env.xRange));
    ly = max(abs(env.yRange));
    lv = env.maxVel;

    psi = NaN(1, Nk);

    k = 1;
    for i_x  = 1:Q

        psi(k:k+Ns-1) = [
            (x/lx),...
            (y/ly),...
            (dx/lv),...
            (dy/lv)
        ].^i_x;

        k = k + Ns;
    end

end