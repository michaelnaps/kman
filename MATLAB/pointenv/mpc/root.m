%% clear workspace
clean;


%% add paths and read in data
addpath ../.
addpath ../../.
addpath ../sphereworld
addpath ./data

load sphereworld_nowall;
Nw = length(world);

run /home/michaelnaps/Downloads/cvx/cvx_setup
clc;

load K_22x22;


%% time parameters
Np = 100;
T = 2;
tspan = (0:dt:T)';
Nt = length(tspan);


%% create test environment
robotRadius = 0.40;
uref = [0, 0];
x0 = [0, -8, 0, 0];
xG = [5, 6, 0, 0];
Nx = length(x0);

observationFun = @(x, u) observables(x, u, Q, world);


%% run simulation
xm = NaN(Nt,Nx);
xm(1,:) = x0;
[uKoop, xKoop, Psi] = KoopmanMPC(xG, x0, Np, K, Q, observationFun, world, robotRadius);

% for i = 1:Nt-1
%     xm(i+1,:) = model(xm(i,:), uKoop(1,:), dt);
% 
%     fprintf("time: %.3f\n", i*dt);
%     fprintf("uKoop: %.3f, %.3f\n", uKoop(1,:));
%     fprintf("xModl: %.3f, %.3f\n\n", xm(i+1,1:Nx/2));
% end


%% plot results
bernard = struct;
bernard.x = xKoop(1,:);
bernard.r = robotRadius;
bernard.color = 'k';

wallsphere = struct;
wallsphere.x = [0,0];
wallsphere.r = -10;
world = [wallsphere, world];

[~] = plot_path(world, bernard, xG, xKoop);
