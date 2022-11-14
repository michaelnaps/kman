function [x1] = model(x, u, dt)

    if nargin < 3
        dt = 1e-3;
    end

    x1 = x + dt*[
        x(2);
        x(4);
        u(1);
        u(2);
    ]';

end