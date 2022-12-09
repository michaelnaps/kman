function [Psi, Nk, INDEX] = observables(x, u, Q, world)

    if nargin < 3
        Q = 3;
    end

    % for tracking of observable placement in other programs
    INDEX = struct;

    Nx = length(x);
    Nw = length(world);
    No = 2*Nw;
    Nu = length(u);
    Nxu = (Nx+Nu)*Nu;

    % for tracking dimensions in other programs
    Nk = Q*Nx + Nw + Nu + Nxu + No + 1;

    % obstacle distances
    dist = NaN(1, Nw);
    for i = 1:Nw
        dist(i) = distance(world(i), [x(1),x(2)]);
    end

    Psi = NaN(1, Nk);

    k = 1;
    for q = 1:Q
        INDEX.("x"+q) = k:k+Nx-1;
        Psi(INDEX.("x"+q)) = x.^q;
        k = k + Nx;
    end

    INDEX.("d") = k:k+Nw-1;
    Psi(INDEX.("d")) = dist.^2;
    k = k + Nw;

    INDEX.("u") = k:k+Nu-1;
    Psi(INDEX.("u")) = u;
    k = k + Nu;

    INDEX.("xu") = k:k+(Nx+Nu)*Nu-1;
    xu = [x, u]'*u;
    Psi(INDEX.("xu")) = xu(:)';
    k = k + (Nx+Nu)*Nu;

    INDEX.("c") = k;
    Psi(INDEX.("c")) = 1;
    k = k + 1;

    for i = 1:Nw
        INDEX.("o"+i) = k:k+1;
        Psi(INDEX.("o"+i)) = world(i).x;
        k = k + 2;
    end


end