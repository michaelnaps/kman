function [xn] = model(x, u, dt)

    if nargin < 3
        dt = 1e-3;
    end

    if nargin < 2
        u = [0,0];
    end

    xn = x + dt*[
        x(3);
        x(4);
        u(1);
        u(2);
    ]';

end