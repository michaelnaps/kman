function [u] = KoopmanMPC(xG, x0, K, Np, Nw, obsFun)

    Ns = length(x0);
    Nu = round(Ns/2);
    [Nk,~] = size(K);

    cvx_begin
        variable u(Nk,Nu)
    cvx_end

end