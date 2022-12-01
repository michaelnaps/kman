clean;


%% add paths and read in data
addpath ../.
addpath ../../.
addpath ../sphereworld
addpath ./data

load sphereworld_minimal world;
Nw = length(world);

load K_11x11


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
bernard.xCenter = x(1,1:2);
bernard.radius = 0.40;
bernard.distInfluence = 0;
bernard.color = [0.4940, 0.1840, 0.5560];

[path_fig] = plot_path(world, bernard, xG, x);

if 0
    figure_path = "/home/michaelnaps/prog/bu_research/literature/koopman_collision_avoidance/figures/";
    exportgraphics(path_fig, figure_path + "prop_environment.png", 'resolution', 600);
end
