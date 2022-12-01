function [u, x] = KoopmanMPC(xg, x0, K, Np, Nw, obsFun)

    Nx = length(x0);
    Nu = round(Nx/2);
    [Nk,~] = size(K);

    Kx = K(:,1:Nx);
%     Ku = K(:,Nx+1:Nx+Nu);
    Kd = K(:,Nx+Nu+1:Nx+Nu+Nw);

    xU = [0,0,0,0];
    uX = [0,0];

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

        variable x(Np, Nx)
        variable u(Np-1, Nu);

        minimize( cost(u,x,xg,Np) );

        subject to

            x(1,:) == x0;

            ku = 1;
            for i = 1:Np-1
    
                x(i+1,:) == obsFun(x(i,:), u(i,:))*Kx;
%                 x(i+1,:) == obsFun(x(i,:), uX)*Kx + obsFun(xU, u(i,:))*Kx;
                ku = ku + Nu;
                
            end

    cvx_end

end

function [C] = cost(u, x, xg, Np)
%     C = u*u' + (x(end,:) - xg)*(x(end,:) - xg)';
    
    C = 0;
    for i = 1:Np-1
        C = C + x(i,:)*x(i,:)' + u(i,:)*u(i,:)';
    end

    C = C + (x(end,:) - xg)*(x(end,:) - xg)';
end