clean;


%% add paths and read in data
addpath ../.
addpath ../../.
addpath ../sphereworld
addpath ./data

load sphereworld_minimal world;

load K_11x11


%% xU and uX
Nt = 1000;
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
Nw = length(world);
Nx = length(x0);
Nu = length(u);
Nk = length(K);


%% koopman operator modification
Kx = K(:,1:Nx);
Ku = K(Nx+1:Nx+Nu,1:Nx);


%% state matrices
xModl = NaN(Nt, Nx);
xKoop = NaN(Nt, Nx);
xPsi = NaN(Nt, Nk);

xModl(1,:) = x0;
xKoop(1,:) = x0;
xPsi(1,:) = Psi0;

for i = 1:Nt-1
    xModl(i+1,:) = obsFun(xModl(i,:), u)*Kx;

    xKoop(i+1,:) = xPsi(i,:)*Kx + u*Ku;
    xPsi(i+1,:) = obsFun(xKoop(i+1,:), [0,0]);
end

disp([xModl, NaN(Nt,1), xPsi(:,1:Nx)])
disp(sum(abs(xModl - xKoop) < TOL, 'all') == Nt*Nx)