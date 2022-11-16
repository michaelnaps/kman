function [xn] = model(x, dt)

    if nargin < 2
        dt = 1e-3;
    end

    c = 1;

    xn = x + dt*[
        x(3);
        x(4);
        0 - c*x(3);
        0 - c*x(4);
    ]';

end