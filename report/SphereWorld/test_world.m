clean;

plot_results = 1;
anim_results = ~plot_results;

addpath ../.
addpath ./sphereworld;

%% generate sphere world wnvironment
load sphereworld;

xGoal = [
    4, 7.25;
    6, 0
];

%% simulation variables
T = 10;  dt = 0.01;
tspan = (0:dt:T)';

%% Test animation function
control  = @(x) (xGoal(:,1)' - x(1:2)) + 5*([0,0] - x(3:4));
modelFun = @(x) model(x, dt, control);

x0 = [0, 0, 0, 0, 0, 0];
x_list = generate_data(modelFun, tspan, x0);

%% animate robot in sphere world
robot = struct;
robot.xCenter = [0;0];
robot.radius = 0.25;
robot.distInfluence = 0.25;
robot.color = [0.4660 0.6740 0.1880];

animate(robot, x_list, tspan, world, xGoal);

