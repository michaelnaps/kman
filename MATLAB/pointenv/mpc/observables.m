function [psi] = observables(x, u, world, Q)

    if nargin < 4
        Q = 3;
    end

    Ns = length(x);
    Nu = length(u);
    Nw = length(world);
    Nk = (Ns + Nu + Nw)*Q + 1;

    xpos = x(1);  xvel = x(3);
    ypos = x(2);  yvel = x(4);

    % obstacle distances
    for i = 1:Nw
        dist(i) = distance(world(i), [xpos, ypos]);
    end

    k = 1;
    for i_x  = 1:Q

        psi(k:k+Ns+Nu+Nw-1) = [
            xpos,...
            ypos,...
            xvel,...
            yvel,...
            u,...
            dist
        ].^i_x;

        k = k + Ns + Nu + Nw;
    end

    psi(end+1) = 1;

end