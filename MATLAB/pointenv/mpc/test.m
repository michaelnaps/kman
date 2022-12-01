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
obsFun = @(x,u) observables(x, u, world, Q);


%% test linear combination
TOL = 1e-6;
x0 = [0,0,0,0];
u  = [0.3,0.5];
Nx = length(x0);
Kx = K(:,1:Nx);

x_stnd = NaN(Nt, Nx);
x_comb = NaN(Nt, Nx);

x_stnd(1,:) = x0;
x_comb(1,:) = x0;

for i = 1:Nt-1
    x_stnd(i+1,:) = obsFun(x_stnd(i,:), u)*Kx;
    x_comb(i+1,:) = obsFun(x_comb(i,:), uX)*Kx + obsFun(xU, u)*Kx;

    disp(sum(x_stnd(i+1,:)-x_comb(i+1,:) < TOL , 'all'))
end

disp([x_stnd, NaN(Nt,1), x_comb])