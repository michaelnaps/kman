function [xn] = model(x, u, dt)

    if nargin < 3
        dt = 1e-2;
    end

    c = 0.75;

    xn = x + dt*[
        u(1);
        u(2)
    ]';

end