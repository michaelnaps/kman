function [xn] = model(x, u, dt)

    if nargin < 3
        dt = 1;
    end

    xn = x + dt*[
        u(1);
        u(2)
    ]';

end