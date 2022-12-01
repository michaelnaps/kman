clean;


%% add paths and read in data
addpath ../.
addpath ../../.
addpath ../sphereworld
addpath ./data

load sphereworld_minimal world;
Nw = length(world);

run /home/michaelnaps/Downloads/cvx/cvx_setup
clc;

load K_11x11


%% time parameters
Np = 10;
T = 30;  Nt = T/dt+1;
tspan = (0:dt:T)';


%% create test environment
xG = [5, 6, 0, 0];
x0 = [0, -8, 0, 0];
Nx = length(x0);

uref = [0, 0];
Nu = length(uref);

observation = @(x,u) observables(x, u, world, Q);
[u, xmpc] = KoopmanMPC(xG, x0, K, Np, Nw, observation);
u = reshape(u, [Nu, Np-1])';

xkoop = NaN(Np, Nx);
xkoop(1,:) = x0;
for i = 1:Np-1
    xkoop(i+1,:) = observation(xkoop(i,:),u(i,:))*K(:,1:Nx);
end


%% run simulation
bernard = struct;
bernard.xCenter = [0,0];
bernard.radius = 0.25;
bernard.distInfluence = 0.25;
bernard.color = 'k';

% [~] = plot_path(world, bernard, xG, xkoop);
