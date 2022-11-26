clean;


%% generate data
addpath ../.
addpath ../../.
addpath ../sphereworld
addpath ./data

load sphereworld world xStart;
Nw = length(world);

load K_11x11


%% time parameters
T = 10;
tspan = (0:dt:T*dt)';


%% create test environment
x0 = [0, -8, 0, 0];
