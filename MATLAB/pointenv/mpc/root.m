%% clear workspace
clean;


%% add paths and read in data
addpath ../.
addpath ../../.
addpath ../sphereworld
addpath ./data

load sphereworld_minimal;
Nw = length(world);

run /home/michaelnaps/Downloads/cvx/cvx_setup
clc;

load K_10x10;


%% time parameters
Np = 100;
T = 2;
tspan = (0:dt:T)';
Nt = length(tspan);


%% create test environment
uref = [0, 0];
x0 = [0, -8, 0, 0];
xG = [5, 6, 0, 0];
Nx = length(x0);

observationFun = @(x, u) observables(x, u, Q, world);


%% run simulation
xm = NaN(Nt,Nx);
xm(1,:) = x0;
[uKoop, xKoop, Psi] = KoopmanMPC(xG, x0, Np, K, Q, observationFun, 0.50);

% for i = 1:Nt-1
%     xm(i+1,:) = model(xm(i,:), uKoop(1,:), dt);
% 
%     fprintf("time: %.3f\n", i*dt);
%     fprintf("uKoop: %.3f, %.3f\n", uKoop(1,:));
%     fprintf("xModl: %.3f, %.3f\n\n", xm(i+1,1:Nx/2));
% end


%% plot results
bernard = struct;
bernard.xCenter = [0,0];
bernard.radius = 0.25;
bernard.distInfluence = 0.25;
bernard.color = 'k';

[~] = plot_path(world, bernard, xG, xKoop);
