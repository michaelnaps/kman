clean;


%% add paths and read in data
addpath ../.
addpath ../../.
addpath ../sphereworld
addpath ./data

load sphereworld_minimal world;
Nw = length(world);

% run /home/michaelnaps/Downloads/cvx/cvx_setup
% clc;

load K_11x11


%% xU and uX
Nt = 100;
xU = [0,0,0,0];
uX = [0,0];
obsFun = @(x, u) observables(x, u, world, Q);


%% test linear combination
TOL = 1e-6;
x0 = [0,0,0,0];
Psi0 = obsFun(x0, [0,0]);
u  = [0.3,0.5];
uPsi = obsFun([0,0,0,0], u);


%% dimension variables
Nx = length(x0);
Nu = length(u);
Nk = length(K);


%% koopman operator modification
Kx = K(:,1:Nx);
Ku = K(:,Nx+1:Nx+Nu);


%% state matrices
x = NaN(Nt, Nx);
xPsi = NaN(Nt, Nk);

x(1,:) = x0;
xPsi(1,:) = Psi0;

for i = 1:Nt-1
    x(i+1,:) = obsFun(x(i,:), u)*Kx;
    xPsi(i+1,:) = xPsi(i,:)*K + uPsi*K;

    disp(sum(x(i+1,:)-xPsi(i+1,1:Nx) < TOL, 'all') == 4)
end

disp([x, NaN(Nt,1), xPsi(:,1:Nx)])