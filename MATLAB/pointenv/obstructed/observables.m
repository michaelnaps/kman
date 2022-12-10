function [Psi, Nk, META] = observables(x, u, Q, world)

    if nargin < 3
        Q = 3;
    end

    % for tracking of observable placement in other programs
    META = struct;

    Nx = length(x);
    Nxx = Nx*Nx;
    Nw = length(world);
    Nu = length(u);
    Nxu = (Nx+Nu)*Nu;
    Nxr = 2*Nx;

    % for tracking dimensions in other programs
    Nk = Nx + Nxx + Nw + Nu + Nxu + 0*Nw*Nxr + 1;

    % obstacle distances
    dist = NaN(1, Nw);
    for i = 1:Nw
        dist(i) = sqrt((x - world(i).x)*(x - world(i).x)');
    end

    Psi = NaN(1, Nk);

    k = 1;

    META.("x") = k:k+Nx-1;
    Psi(META.("x")) = x;
    k = k + Nx;

    META.("xx") = k:k+Nxx-1;
    xx = x'*x;
    Psi(META.("xx")) = xx(:);
    k = k + Nxx;

    META.("u") = k:k+Nu-1;
    Psi(META.("u")) = u;
    k = k + Nu;

    META.("xu") = k:k+(Nx+Nu)*Nu-1;
    xu = [x, u]'*u;
    Psi(META.("xu")) = xu(:)';
    k = k + (Nx+Nu)*Nu;

%     for i = 1:Nw
%         r = world(i).x;
%         xr = x'*r;
% 
%         META.("xr"+i) = k:k+Nxr-1;
%         Psi(META.("xr"+i)) = xr(:)';
%         
%         k = k + Nxr;
%     end

    META.("d") = k:k+Nw-1;
    Psi(META.("d")) = dist;
    k = k + Nw;

    META.("c") = k;
    Psi(META.("c")) = 1;
%     k = k + 1;

end