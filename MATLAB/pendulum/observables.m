function [psi] = observables(x, Q)

    if nargin < 2
        Q = 3;
    end

    Nk = 2*Q^2 + 2;

    th = x(1);  dth = x(2);
    l1 = 2*pi;  l2  = 1;

    psi = NaN(1,Nk);
    psi(1:2) = [x(1), x(2)];
    k = 3;
    for i = 1:Q

        for j = 1:Q

%             for p = 1:Q
            
                psi(k:k+1) = [cos(th)^i, (dth/l2)^j];
                k = k + 2;

%             end

        end

    end

end