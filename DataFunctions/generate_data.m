function [xlist] = generate_data(F, tspan, x0, ulist, Nu)
    
    % inputs: F, tspan, x0, ulist
    % where function, F, is a discrete dynamics function
    
    [Nx, N0] = size(x0);
    Nt = length(tspan);
    
    if nargin < 5
        Nu = Nx;
    end
    
    xlist = NaN(N0*Nx, Nt);

    n = 1;
    k = 1;
    for i = 1:N0

        x = NaN(Nx, Nt);
        x(:,1) = x0(:,i);

        for t = 1:Nt-1
          
            x(:,t+1) = F(x(1:Nx,t), ulist(k:k+Nu-1,t));

        end

        xlist(n:n+Nx-1,:) = x;
        n = n + Nx;
        k = k + Nu;

    end

end