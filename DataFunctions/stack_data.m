function [x] = stack_data(data, Nx, Ns, Nt)
    
    x = NaN(Ns, Nx*Nt);

    k = 1;
    t = 1;
    for i = 1:Nx
        x(:,t:t+Nt-1) = data(k:k+Ns-1,:);

        k = k + Ns;
        t = t + Nt;
    end

end