function [Psi] = observables(x, u, Q, world)

    if nargin < 3
        Q = 3;
    end

    Nx = length(x);
    Nu = length(u);
    Nw = length(world);
    Nk = Nx*Q + Nw + Nu + (Nx+Nu)*Nx + 1;

    % obstacle distances
    dist = NaN(1, Nw);
    for i = 1:Nw
        dist(i) = distance(world(i), [x(1),x(2)]);
    end

    Psi = NaN(1, Nk);

    k = 1;
    for q = 1:Q
        Psi(k:k+Nx-1) = x.^q;
        k = k + Nx;
    end

    Psi(k:k+Nw-1) = dist.^2;
    k = k + Nw;

    Psi(k:k+Nu-1) = u;
    k = k + Nu;

    xu = [x, u]'*u;
    Psi(k:k+(Nx+Nu)*Nu-1) = xu(:)';
    k = k + (Nx+Nu)*Nu;

    Psi(k) = 1;

end