function [xlist] = generate_data(F, tspan, x0, ulist, Nu)
    
    % inputs: F, tspan, x0, ulist
    % where function, F, is a discrete dynamics function
    
    [N0, Nx] = size(x0);
    Nt = length(tspan);
    
    if nargin < 5
        Nu = Nx;
    end
    
    xlist = NaN(Nt, N0*Nx);

    n = 1;
    k = 1;
    for i = 1:N0

        x = NaN(Nt, Nx);
        x(1,:) = x0(i,:);

        for t = 1:Nt-1
          
            x(t+1,:) = F(x(t,1:Nx), ulist(t,k:k+Nu-1));

        end

        xlist(:,n:n+Nx-1) = x;
        n = n + Nx;
        k = k + Nu;

    end

end