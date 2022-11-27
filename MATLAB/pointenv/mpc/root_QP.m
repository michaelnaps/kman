clean;


%% add paths and read in data
addpath ../.
addpath ../../.
addpath ../sphereworld
addpath ./data

load sphereworld world;
Nw = length(world);

load K_19x19


%% time parameters
T = 30;  Nt = T/dt + 1;
tspan = (0:dt:T)';


%% create test environment
x0 = [0, -8, 0, 0];
xG = [5, 6];
Nx = length(x0);

controlledModel = @(x) model(x, ModelMPC(xG, x, world), dt);


%% run simulation
x = NaN(Nt, 2);
x(1,:) = x0(1:2);
for i = 2:Nt
    x(i,:) = x(i-1,:) + dt*ModelQP(xG, x(i-1,:), world);
end

bernard = struct;
bernard.xCenter = [0,0];
bernard.radius = 0.25;
bernard.distInfluence = 0.25;
bernard.color = 'k';

[~] = plot_path(bernard, x, world, xG);
