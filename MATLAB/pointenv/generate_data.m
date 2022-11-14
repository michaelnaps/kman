function [x_list] = generate_data(F, tspan, x0)

    % where function, F, is a discrete dynamics function
    
    [Nx, Ns] = size(x0);
    Nt = length(tspan);    % number of time-steps

    x_list = NaN(Nt, Nx*Ns);

    n = 1;
    for i = 1:Nx

        x = NaN(Nt, Ns);
        x(1,:) = x0(i,:);

        for t = 1:Nt-1
            
            x(t+1,:) = F(x(t,:));

        end

        x_list(:,n:n+Ns-1) = x;
        n = n + Ns;

    end

end