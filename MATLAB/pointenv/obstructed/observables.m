function [Psi] = observables(x, u, Q, world)

    if nargin < 3
        Q = 3;
    end

    Ns = length(x);
    Nu = length(u);
    Nw = length(world);
    Nk = Ns*Q + Nw + Nu;

    % obstacle distances
    dist = NaN(1, Nw);
    for i = 1:Nw
        dist(i) = distance(world(i), [x(1), x(2)]);
    end

    Psi = NaN(1, Nk);

    k = 1;
    for i = 1:Q
        Psi(k:k+Ns-1) = x.^i;
        k = k + Ns;
    end

    Psi(k:k+Nw-1) = dist;
    k = k + Nw;

    Psi(k:k+Nu-1) = u;

end