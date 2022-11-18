function [psi] = observables(x, u, Q, env)

    if nargin < 3
        env = struct;
        env.xRange = [0, 1];
        env.yRange = [0, 1];
        env.maxVel = 1;
    end

    if nargin < 2
        Q = 3;
    end

    Ns = length(x);
    Nk = Ns*Q;

    xpos = x(1);  xvel = x(3);
    ypos = x(2);  yvel = x(4);

    ux = u(1);  uy = u(2);

    lx = max(abs(env.xRange));
    ly = max(abs(env.yRange));
    lv = env.maxVel;

    psi = NaN(1, Nk);

    k = 1;
    for i_x  = 1:Q

        psi(k:k+Ns-1) = [
            (xpos/lx),...
            (ypos/ly),...
            (xvel/lv),...
            (yvel/lv),...
            u
        ].^i_x;

        k = k + Ns;
    end

end