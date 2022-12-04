clean;


%% add paths and read in data
addpath ../.
addpath ../../.
addpath ../sphereworld
addpath ./data

load sphereworld_minimal world;

load K_19x19


%% xU and uX
Nt = 1000;
xU = [0,0,0,0];
uX = [0,0];
obsFun = @(x, u) observables(x, u, world, Q);


%% test linear combination
TOL = 1e-6;
x0 = [0,0,0,0];
Psi0 = obsFun(x0, [0,0]);
u  = [0.3, 0.5];


%% dimension variables
Nw = length(world);
Nx = length(x0);
Nu = length(u)/2;
Nk = length(K);


%% model propagation functions
modelFun = @(x,u) model(x, u, dt);
Kx = K(1:Q*Nx,1:Q*Nx);
Ku = K(end-Nu-1:end-1,1:Q*Nx);


%% state matrices
xModl = NaN(Nt, Nx);
xKoop = NaN(Nt, Q*Nx);
xPsi = NaN(Nt, Nk);

xModl(1,:) = x0;
xKoop(1,:) = Psi0(1:Q*Nx);

for i = 1:Nt-1

    xModl(i+1,:) = modelFun(xModl(i,:), u);
    xKoop(i+1,:) = xKoop(i,:)*Kx + u*Ku;

end

disp([xModl, NaN(Nt,1), xKoop(:,1:Nx)])
disp(sum(abs(xModl - xKoop(:,1:Nx)) < TOL, 'all') == Nt*Nx)


















