function [psi] = observables(x, u, Q)

    if nargin < 3
        Q = 3;
    end

    Ns = length(x);
    Nu = length(u);
    Nk = Ns*Q + Nu;

    psi = NaN(1, Nk);

    k = 1;
    for i = 1:Q
        psi(k:k+Ns-1) = x.^i;
        k = k + Ns;
    end

    psi(k:k+Nu-1) = u;

end