function [Psi, META] = observables(x, u, world)
    META = struct;

    Nx = length(x);
    Nxx = Nx*Nx;
    Nu = length(u);
    Nuu = Nu*Nu;
    Nxu = Nx*Nu;
    Nw = length(world);

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

    META.("uu") = k:k+Nuu-1;
    uu = u'*u;
    Psi(META.("uu")) = uu(:);
    k = k + Nuu;

    META.("xu") = k:k+Nxu-1;
    xu = x'*u;
    Psi(META.("xu")) = xu(:);
    k = k + Nxu;

    META.("ux") = k:k+Nxu-1;
    ux = u'*x;
    Psi(META.("ux")) = ux(:);
    k = k + Nxu;

    d = NaN(1,Nw);
    for i = 1:Nw
        d(i) = (x - world(i).x)*(x - world(i).x)';
    end

    META.("d") = k:k+Nw-1;
    Psi(META.("d")) = d;
    k = k + Nw;

    META.Nk = k - 1;

    META.labels = [
        "x_1", "x_2", "x_1x_1", "x_1x_2", "x_2x_1", "x_2x_2",...
        "u_1", "u_2", "u_1u_1", "u_1u_2", "u_2u_1", "u_2u_2",...
        "x_1u_1", "x_2u_1", "x_1u_2", "x_2u_2",...
        "x_1u_1", "x_1u_2", "x_2u_1", "x_2u_2",...
        "o_W", "o_1", "o_2", "o_3",...
    ];

end