function [x_list] = generate_data(F, tspan, x0, inputFun)

    if nargin < 4
        inputFun = @(x) [];
    end

    % where function, F, is a discrete dynamics function
    
    [Nx, Ns] = size(x0);
    Nt = length(tspan);    % number of time-steps
    Nu = length(inputFun(x0(1,:)));
    
    x_list = NaN(Nt, Nx*(Ns + Nu));

    n = 1;
    for i = 1:Nx

        x = NaN(Nt, Ns+Nu);
        x(1,:) = [x0(i,:), zeros(1,Nu)];

        for t = 1:Nt-1
            
            u = inputFun(x(t,:));
            x(t+1,:) = [F(x(t,1:Ns), u), u];

        end

        x_list(:,n:n+Ns+Nu-1) = x;
        n = n + (Ns+Nu);

    end

end