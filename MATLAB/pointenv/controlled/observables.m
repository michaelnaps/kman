function [Psi, Nk, META] = observables(x, u)
    % for tracking of observable placement in other programs
    META = struct;

    Nx = length(x);
    Nxx = Nx*Nx;
    Nu = length(u);
    Nxu = (Nx+Nu)*Nu;

    % for tracking dimensions in other programs
    Nk = Nx + Nxx + Nu + Nxu + 1;
    META.Nk = Nk;

    k = 1;
    Psi = NaN(1, Nk);

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

    META.("c") = k;
    Psi(META.("c")) = 1;
%     k = k + 1;

end