function [xn] = model(x, dt, inputFun)

    if nargin < 3
        inputFun = @(x) [];
    end

    if nargin < 2
        dt = 1e-3;
    end

    xn = x + dt*[
        x(3);
        x(4);
        x(5);
        x(6);
        inputFun(x(1:4))'        
    ]';

end