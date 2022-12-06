function [u] = ModelMPC(Np, x0, xG, uref, world)
    
    Nx = length(x0);
    Nu = length(uref);
    Nw = length(world);

%     Q = eye(Nx);
%     P = eye(Nu);
%     R = eye(Nx);

    H = eye(Np*(Nx+Nu)+Nx);
    f = zeros(Np*(Nx+Nu)+Nx, 1);

    x = NaN(1, Np*Nx);
    x(1:4) = x0;

    p = 5;
    for i = 2:Np

       x(p:p+Nx-1) = model(x(p-Nx:p-1), uref);
       p = p + Nx;

    end

    u = [0,0];

end