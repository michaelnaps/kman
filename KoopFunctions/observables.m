function [Psi, meta] = observables(x, u, world)
    meta = struct;

    Nx = length(x);
    Nxx = Nx*Nx - 1;
    Nu = length(u);
    Nuu = Nu*Nu - 1;
    Nxu = Nx*Nu;
%     Nux = Nu*Nx;
    Nw = length(world);

    k = 1;

    meta.("x") = k:k+Nx-1;
    Psi(meta.("x")) = x;
    k = k + Nx;

    meta.("xx") = k:k+Nxx-1;
    xx = x'*x;
%     Psi(meta.("xx")) = xx(:);
    Psi(meta.("xx")) = [xx(1), xx(2), xx(4)];
    k = k + Nxx;

    meta.("u") = k:k+Nu-1;
    Psi(meta.("u")) = u;
    k = k + Nu;

    meta.("uu") = k:k+Nuu-1;
    uu = u'*u;
%     Psi(meta.("uu")) = uu(:);
    Psi(meta.("uu")) = [uu(1), uu(2), uu(4)];
    k = k + Nuu;

    meta.("xu") = k:k+Nxu-1;
    xu = x'*u;
    Psi(meta.("xu")) = xu(:);
    k = k + Nxu;

%     meta.("ux") = k:k+Nxu-1;
%     ux = u'*x;
%     Psi(meta.("ux")) = ux(:);
%     k = k + Nxu;

    d = NaN(1,Nw);
    for i = 1:Nw
        d(i) = (x - world(i).x)*(x - world(i).x)';
    end

    meta.("d") = k:k+Nw-1;
    Psi(meta.("d")) = d;
    k = k + Nw;

    meta.("c") = k;
    Psi(meta.("c")) = 1;
    meta.Nk = k;

%     meta.labels = [
%         "q_x", "q_y", "q_xq_x", "q_xq_y", "q_yq_x", "q_yq_y",...
%         "u_x", "u_y", "u_xu_x", "u_xu_y", "u_yu_x", "u_yu_y",...
%         "q_xu_x", "q_yu_x", "q_xu_y", "q_yu_y",...
%         "q_xu_x", "q_xu_y", "q_yu_x", "q_yu_y",...
%         "o_W", "o_1", "o_2", "o_3",...
%     ];

    meta.labels = [
        "x_1", "x_2", "x_1x_1", "x_1x_2", "x_2x_2",...
        "u_1", "u_2", "u_1u_1", "u_1u_2", "u_2u_2",...
        "x_1u_1", "x_2u_1", "x_1u_2", "x_2u_2",...
        ..."u_1x_1", "u_2x_1", "u_1x_2", "u_2x_2",...
        "o_W", "o_1", "o_2", "o_3", "c"...
    ];

end