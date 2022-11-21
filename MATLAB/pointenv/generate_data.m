function [xlist] = generate_data(F, tspan, x0, ulist)
    
    % inputs: F, tspan, x0, ulist
    % where function, F, is a discrete dynamics function
    
    [Nx, Ns] = size(x0);
    Nu = round(Ns/2);
    Nt = length(tspan);
    
    xlist = NaN(Nt, Nx*Ns);

    n = 1;
    k = 1;
    for i = 1:Nx

        x = NaN(Nt, Ns);
        x(1,:) = x0(i,:);

        for t = 1:Nt-1
          
            x(t+1,:) = F(x(t,1:Ns), ulist(t,k:k+Nu-1));

        end

        xlist(:,n:n+Ns-1) = x;
        n = n + Ns;
        k = k + Nu;

    end

end