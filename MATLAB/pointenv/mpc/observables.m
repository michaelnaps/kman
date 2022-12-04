function [psi] = observables(x, u, world, Q)
    if nargin < 4
        Q = 3;
    end

    Nx = length(x);
    Nu = length(u);
    Nw = length(world);
    % Nk = (Ns + Nu + Nw)*Q + 1;

    % obstacle distances
    for i = 1:Nw
        dist(i) = distance(world(i), [x(1), x(2)]);
    end

    k = 1;
    for q = 1:Q

        psi(k:k+Nx-1) = x.^q;
        k = k + Nx;

    end

    for q = 1:Q
        
        psi(k:k+Nu-1) = u.^q;
        k = k + Nu;

    end

    for q = 1:Q

        psi(k:k+Nw-1) = dist.^q;
        k = k + Nw;

    end

    psi(end+1) = 1;

end