function [Psi] = observables(x, u, Q, world)

    if nargin < 3
        Q = 3;
    end

    Nx = length(x);
    Nu = length(u);
    Nw = length(world);
    Nk = (Nx + 2*Nw)*Q + Nu;

    % obstacle distances
    dist = NaN(1, 2*Nw);

    j = 1;
    for i = 1:Nw
        dist(j:j+1) = (world(i).xCenter' - x(1:2));
        j = j + 2;
    end

    Psi = NaN(1, Nk);

    k = 1;
    for q = 1:Q
        Psi(k:k+Nx-1) = x.^q;
        k = k + Nx;
    end

    for q = 1:Q
        Psi(k:k+2*Nw-1) = dist.^q;
        k = k + 2*Nw;
    end

    Psi(k:k+Nu-1) = u;

end