function [xn] = model(x, u, alpha)

    if nargin < 3
        alpha = 1;
    end

    xn = x + alpha*[
        u(1);
        u(2)
    ]';

end