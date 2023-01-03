function [dx] = fdm(f, x, h)
    if nargin < 3
        h = 1e-3;
    end

    Nr = length(f(x));
    Nc = length(x);
    dx = NaN(Nr, Nc);

    for i = 1:Nc
        xn2 = x;  xn2(i) = x(i) - 2*h;
        xn1 = x;  xn1(i) = x(i) - h;
        xp1 = x;  xp1(i) = x(i) + h;
        xp2 = x;  xp2(i) = x(i) + 2*h;

        yn2 = f(xn2);
        yn1 = f(xn1);
        yp1 = f(xp1);
        yp2 = f(xp2);

        dx(:,i) = (yn2 - yp2 + 8*yp1 - 8*yn1)/(12*h);
    end

    dx = dx';
end
