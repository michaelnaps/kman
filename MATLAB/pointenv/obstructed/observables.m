function [Psi] = observables(x, u, Q, world)

    if nargin < 3
        Q = 3;
    end

    Nx = length(x);
    Nu = length(u);
    Nw = length(world);
    Nk = (Nx + Nw)*Q + Nu;

    % obstacle distances
    dist = NaN(1, Nw);
    for q = 1:Nw
        dist(q) = distance(world(q), [x(1), x(2)]);
    end

    Psi = NaN(1, Nk);

    k = 1;
    for q = 1:Q
        Psi(k:k+Nx-1) = x.^q;
        k = k + Nx;
    end

    for q = 1:Q
        Psi(k:k+Nw-1) = dist.^q;
        k = k + Nw;
    end

    Psi(k:k+Nu-1) = u;

end