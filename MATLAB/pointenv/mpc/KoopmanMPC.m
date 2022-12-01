function [u, x] = KoopmanMPC(xg, x0, K, Np, Nw, obsFun)

    [Nk, ~] = size(K);
    Nx = length(x0);
    Nu = round(Nx/2);

    Kx = K(:,1:Nx);
%     Ku = K(:,Nx+1:Nx+Nu);
%     Kd = K(:,Nx+Nu+1:Nx+Nu+Nw);

    Psig = obsFun(xg, [0,0]);
    Psi0 = obsFun(x0, [0,0]);

    cvx_begin
        variable Psi(Np, Nk);
        variable x(Np, Nx);
        variable u(Np-1, Nu);

        minimize( cost(u,Psi,Psig,Np) );

        subject to
            xk(1,:) == Psi0;
            
            for i = 1:Np-1
                xk(i+1,:) == Psi(i,:)*K(:,1:Nx) + Psi(i,:)*K(:,1:Nx);
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