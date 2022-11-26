function [psi] = observables(x, u, world, Q, env)

    if nargin < 5
        env = struct;
        env.xRange = [0, 1];
        env.yRange = [0, 1];
        env.maxVel = 1;
    end

    if nargin < 4
        Q = 3;
    end

    Ns = length(x);
    Nu = length(u);
    Nw = length(world);
    Nk = (Ns + Nu + 3*Nw)*Q + 1;

    xpos = x(1);  xvel = x(3);
    ypos = x(2);  yvel = x(4);

    lx = max(abs(env.xRange));
    ly = max(abs(env.yRange));
    lv = env.maxVel;

    % obstacle distances
    dist = NaN(1, Nw);
    grad = NaN(Nw, 2);

    for i = 1:Nw
        dist(i) = distance(world(i), [xpos, ypos]);
        grad(i,:) = distance_grad(world(i), [xpos, ypos], dist(i));
    end

    psi = NaN(1, Nk);

    k = 1;
    for i_x  = 1:Q

        psi(k:k+Ns+Nu+3*Nw-1) = [
            (xpos/lx),...
            (ypos/ly),...
            (xvel/lv),...
            (yvel/lv),...
            u,...
            dist,...
            reshape(grad, [1, 2*Nw])
        ].^i_x;

        k = k + Ns + Nu + 3*Nw;
    end

    psi(end) = 1;

end