function [u, x] = KoopmanMPC(xg, x0, K, Np, Nw, obsFun)

    Nx = length(x0);
    Nu = round(Nx/2);
    [Nk,~] = size(K);

    Kx = K(:,1:Nx);
%     Ku = K(:,Nx+1:Nx+Nu);
    Kd = K(:,Nx+Nu+1:Nx+Nu+Nw);

    % create goal vector for cost function
    xG = NaN(1,Np*Nx);
    k = 1;
    for i = 1:Np

        xG(k:k+Nx-1) = xg;
        k = k + Nx;

    end

    cvx_begin

%         expression x(1,Nx*Np);
        expressions xs(1,Nx) xi(1,Nx) xf(1,Nx)
        expression u(1,Nu*(Np-1));

        minimize( u*u' + (xf - xg)*(xf - xg)' );

        subject to
            xs(1:Nx) == x0;

            xi == obsFun(xs, u(1:2))*Kx;
            xf == obsFun(xi, u(3:4))*Kx;

%             kx = 1;
%             kxn = kx + Nx;
%             ku = 1;
%             for i = 1:Np-1
%     
%                 x(kxn:kxn+Nx-1) == obsFun(x(kx:kx+Nx-1), u(ku:ku+Nu-1))*Kx;
%                 
%                 kx = kx + Nx;
%                 kxn = kxn + Nx;
%                 ku = ku + Nu;
%                 
%             end

    cvx_end

    u'
    x = [xs, xi, xf];
    x'

end