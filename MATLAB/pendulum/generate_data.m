function [x_list] = generate_data(F, tspan, x0)

    % where function, f, is a discrete dynamics function

    N = length(x0(:,1));  % number of data sets
    M = length(x0(1,:));  % number of state space variables
    K = length(tspan);    % number of time-steps

    x_list = NaN(K, N*M);

    n = 1;
    for i = 1:N

        x = NaN(K, M);
        x(1,:) = x0(i,:);

        for k = 1:K-1
            
            x(k+1,:) = F(x(k,:));

        end

        x_list(:,n:n+M-1) = x;
        n = n + M;

    end

end