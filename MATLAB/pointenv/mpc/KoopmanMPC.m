function [u] = KoopmanMPC(xG, x0, K, Np, Nw, observableFunc)

    Ns = length(x0);
    Nu = round(Ns/2);
    Nk = length(K);

    psi = NaN(Np, Nk);
    psi(1,:) = observableFunc(x0, uD);

    for i = 2:Np

        psi(i,:) = (K'*psi(i-1,:)')';

    end

    x = psi(:,1:Ns+Nu+Nw);

    

end