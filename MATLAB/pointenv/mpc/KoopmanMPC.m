function [u, x] = KoopmanMPC(xg, x0, K, Np, Nw, obsFun)

    Nx = length(x0);
    Nu = round(Nx/2);
    [Nk,~] = size(K);

    Kx = K(:,1:Nx);
%     Ku = K(:,Nx+1:Nx+Nu);
    Kd = K(:,Nx+Nu+1:Nx+Nu+Nw);

%     % create goal vector for cost function
%     xG = NaN(1,Np*Nx);
%     k = 1;
%     for i = 1:Np
% 
%         xG(k:k+Nx-1) = xg;
%         k = k + Nx;
% 
%     end

    cvx_begin

        expression x(Np, Nx)
        variable u(1,Nu*(Np-1));

        minimize( u*u' + (x(end,:) - xg)*(x(end,:) - xg)' );

        subject to

            x(1,:) == x0;

            ku = 1;
            for i = 1:Np-1
    
                x(i+1,:) == obsFun(x(i,:), u(ku:ku+Nu-1))*Kx;
                ku = ku + Nu;
                
            end

    cvx_end

end