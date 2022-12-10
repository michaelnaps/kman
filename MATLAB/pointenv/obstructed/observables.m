function [Psi, Nk, INDEX] = observables(x, u, Q, world)
    % for tracking of observable placement in other programs
    INDEX = struct;

    % dimension variables
    Nx = length(x);
    Nw = length(world);
    No = 2*Nw;
    Nu = length(u);
    Nxu = (Nx+Nu)*Nu;
    Nxuo = (Nx + Nu)*2;

    % for tracking dimensions in other programs
    Nk = Q*Nx + Nw + Nu + Nxu + 0*No + 0*Nw*Nxuo + 1;

    % obstacle distances
    dist = NaN(1, Nw);
    for i = 1:Nw
%         dist(i) = (x - world(i).x)*(x - world(i).x)';
        dist(i) = distance(world(i), [x(1),x(2)]);
    end

    Psi = NaN(1, Nk);

    k = 1;
    for q = 1:Q
        INDEX.("x"+q) = k:k+Nx-1;
        Psi(INDEX.("x"+q)) = x.^q;
        k = k + Nx;
    end

    INDEX.("u") = k:k+Nu-1;
    Psi(INDEX.("u")) = u;
    k = k + Nu;

    INDEX.("xu") = k:k+(Nx+Nu)*Nu-1;
    xu = [x, u]'*u;
    Psi(INDEX.("xu")) = xu(:)';
    k = k + (Nx+Nu)*Nu;

    INDEX.("d") = k:k+Nw-1;
    Psi(INDEX.("d")) = dist.^2;
    k = k + Nw;

%     for i = 1:Nw
%         o = world(i).x;
% 
%         INDEX.("o"+i) = k:k+1;
%         Psi(INDEX.("o"+i)) = o;
%         k = k + 2;
%         
%         INDEX.("oxuo"+i) = k:k+Nxuo-1;
%         oxuo = [x, u]'*o;
%         Psi(INDEX.("oxuo"+i)) = oxuo(:);
%         k = k + Nxuo;
%     end

    INDEX.("c") = k;
    Psi(INDEX.("c")) = 1;

end