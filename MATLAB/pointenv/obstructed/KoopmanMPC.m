function [u] = KoopmanMPC(x0, obsFun, K, Np, Nw, u_def)

    Ns = length(x0);
    Nu = round(Ns/2);
    Nk = length(K);

    if nargin < 4
        u_def = zeros(Np, Nu);
    end

    psi = NaN(Np, Nk);
    psi(1,:) = obsFun(x0, u_def);

    x = NaN(Np, Ns+Nu+Nw);

    for i = 2:Np

        psi(i,:) = (K'*psi(i-1,:)')';

    end

    x = psi(:,1:Ns+Nu+Nw);

end