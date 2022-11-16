function [psi] = observables(q, inputFun, Q, env)

    if nargin < 4
        env = struct;
        env.xRange = [0, 1];
        env.yRange = [0, 1];
        env.maxVel = 1;
    end

    if nargin < 3
        Q = 3;
    end

    Nx = length(q) - 2;
    Nk = Nx*Q^Nx + 2;

    u = inputFun(q);

    x  = q(1);  dx = q(2);
    y  = q(3);  dy = q(4);
    ux = u(1);  uy = u(2);
%     lx = max(abs(env.xRange));
%     ly = max(abs(env.yRange));
    lv = env.maxVel;

%     o1 = x(5);
%     o2 = x(6);
%     o3 = x(7);

    psi = NaN(1, Nk);
    psi(1:2) = [x, y];
%     psi(1:Nx) = [x, dx, y, dy, ux, uy];

    k = 3;
%     for i_x  = 1:Q
%     for i_y  = 1:Q
        for i_dx = 1:Q
        for i_dy = 1:Q
            for i_ux = 1:Q
            for i_uy = 1:Q

                psi(k:k+Nx-1) = [
%                     (x/lx)^i_x;
%                     (y/ly)^i_y;
                    (dx/lv)^i_dx;
                    (dy/lv)^i_dy;
                    ux^i_ux;
                    uy^i_uy
                ]';

                k = k + Nx;

            end
            end
        end
        end
%     end
%     end

end