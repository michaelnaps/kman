% function root(plot_results)
% 
% if nargin < 1
%     plot_results = 1;
% end

clean;

plot_results = 1;
anim_results = ~plot_results;

addpath ../.
addpath ../../.
addpath ../sphereworld;

load sphereworld world xStart;
Nw = length(world);


%% Model function
dt = 0.01;
modelFun = @(x, u) model(x, u, dt);


%% Initialize training data
Nrand = 20;
x0 = [
    xStart', zeros(size(xStart'));
    20*rand(Nrand, 2), 10*rand(Nrand, 2) - 5;
    0, 0, 20, 10
];
[Nx, Ns] = size(x0);
Nu = round(Ns/2);

% simulation variables
T = 10;
tspan = 0:dt:T;
Nt = length(tspan);

% create list of inputs
u0 = 20*rand(Nx, Nu) - 10;
u_generate = NaN(Nt, Nx*Nu);

k = 1;
for i = 1:Nx
    u_generate(:,k:k+Nu-1) = [
        linspace(u0(i,1),0,Nt)', linspace(u0(i,2),0,Nt)'
    ];
    k = k + Nu;
end

% generate model data
data_train = generate_data(modelFun, tspan, x0, u_generate);
x_train = stack_data(data_train, Nx, Ns, Nt);
u_train = stack_data(u_generate, Nx, Nu, Nt);


%% Evaluate for the observation function
Q = 1;

observation = @(x, u) observables(x, u, world, Q);
Nk = length(observation([0,0,0,0], [0,0]));

[K, acc, ~, ~] = KoopmanWithControl(observation, x_train, x0, u_train);
fprintf("L-2 norm: %.3s\n\n", acc)


%% test koopman operator on new data
% time variables
T_koop = 20;
t_koop = (0:dt:T_koop)';
Nt = length(t_koop);

% introduce variance into the initial conditions
x0 = x0(Nx-4:end,:);
[Nx, Ns] = size(x0);
x0 = x0 + [(rand(Nx-1, Ns) - 0.5); zeros(1, Ns)];

% create list of inputs
u0 = 20*rand(Nx, Nu) - 10;
u_test = NaN(Nt, Nx*Nu);
Nl = round(Nt/4);
N0 = Nt - Nl;

k = 1;
for i = 1:Nx
    u_test(:,k:k+Nu-1) = [
        linspace(u0(i,1),0,Nl)', linspace(0,u0(i,2),Nl)';
        zeros(N0, Nu);
    ];
    k = k + Nu;
end


%% generate data for new initial conditions
KoopModel = @(x, u) KoopFun(x, u, world, K, Q);
x_koop = generate_data(KoopModel, t_koop, x0, u_test);
x_test = generate_data(modelFun, t_koop, x0, u_test);


%% generate obstacle distance comparison data
obs_koop = NaN(Nt, Nw);
obs_test = NaN(Nt, Nw);

xc = x0(1,:);

for i = 1:Nt
    psi = observation(xc, u_test(i,1:Nu));
    psi = (K'*psi')';

    obs_koop(i,:) = psi(Nx+Nu:Nx+Nu+Nw-1);

    xc = psi(1:Ns);

    obs_test(i,:) = ModelToSphere(x_test(i,1:Ns), world);
end


%% plot results
if ~isnan(acc)

    if plot_results
        
        plot_comparisons(x_test, x_koop, x0, t_koop);
        plot_comparisons(obs_test, obs_koop, obs_test(1,:), t_koop);

    elseif anim_results

        bernard = struct;
        bernard.xCenter = [0,0];
        bernard.radius = 0.25;
        bernard.distInfluence = 0.25;
        bernard.color = 'k';

        x_test_anim = x_test(:,end-(Ns-1):end);
        x_koop_anim = x_koop(:,end-(Ns-1):end);

        animate(bernard, x_koop_anim, tspan, world, [0,0], x_test_anim);

    end

%     keyboard;

end

% end


%% Functions for modeling comparisons
function [x_n] = KoopFun(x, u, world, K, Q)
    Nx = length(x);
    Nw = length(world);

    psi = observables(x, u, world, Q);
    psi_n = (K'*psi')';
    x_n = psi_n(1:Nx);
end

function [o] = ModelToSphere(x, world)
    Nw = length(world);

    o = NaN(1,Nw);

    for i = 1:Nw
        o(i) = distance(world(i), [x(1), x(2)]);
    end
end










