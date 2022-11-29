clean;


%% add paths and read in data
addpath ../.
addpath ../../.
addpath ../sphereworld
addpath ./data

load sphereworld world;
Nw = length(world);

load K_11x11


%% time parameters
Np = 10;
T = 30;  Nt = T/dt + 1;
tspan = (0:dt:T)';


%% create test environment
x0 = [0, -8, 0, 0];
xG = [5, 6];
Nx = length(x0);
uref = [0, 0];


%% run simulation
% xm = NaN(Nt, Nx);
% xm(1,:) = x0;
% for i = 2:Nt
%     xm(i,:) = controlledModel(xm(i-1,:));
% end
% 
% bernard = struct;
% bernard.xCenter = [0,0];
% bernard.radius = 0.25;
% bernard.distInfluence = 0.25;
% bernard.color = 'k';
% 
% [~] = plot_path(bernard, xm, world, xG);
