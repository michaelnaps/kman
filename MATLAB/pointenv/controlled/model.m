function [xn] = model(x, u, dt)

    if nargin < 2
        dt = 1e-3;
    end

    c = 1;

    xn = x + dt*[
        x(3);
        x(4);
        u(1) - c*x(3);
        u(2) - c*x(4);
    ]';

end