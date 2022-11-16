function [xn] = model(x, dt, inputFun)

    if nargin < 3
        inputFun = @(x) [0,0];
    end

    if nargin < 2
        dt = 1e-3;
    end

    c = 1;

    xn = x + dt*[
        x(3);
        x(4);
        x(5) - c*x(3);
        x(6) - c*x(4);
        inputFun(x(1:4))'
    ]';

end