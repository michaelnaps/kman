clean;

plot_results = 1;
anim_results = ~plot_results;

addpath ../.
addpath ../../.
addpath ../sphereworld;

load sphereworld;


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
Nk = Ns*Q;

observation = @(x, u) observables(x, u, Q);

[K, acc, ind, err] = KoopmanWithControl(observation, x_train, x0, u_train);
fprintf("L-2 norm: %.3s\n\n", acc)


%% test koopman operator on new data
% time variables
T_koop = 20;
t_koop = (0:dt:T_koop)';
Nt = length(t_koop);

% introduce variance into the initial conditions
x0 = x0(Nx-4:end,:);
[Nx, Ns] = size(x0);
x0 = x0 + [(rand(Nx-1, Ns) - 0.5); 0, 0, 0, 0];

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

% psi0 = NaN(Nx, Nk);
% for i = 1:Nx
% 
%     psi0(i,:) = observation(x0(i,:), [0,0]);
% 
% end

koop = @(x, u) KoopFun(x, u, K, Q);
x_koop = generate_data(koop, t_koop, x0, u_test);

% % delete unwanted elements from the observation space
% k = 1;  j = 1;
% x_koop = NaN(Nt,Ns*Nx);
% for i = 1:Nx
% 
%     x_koop(:,j:j+Ns-1) = data_koop(:,k:k+Ns-1);
% 
%     k = k + Nk;
%     j = j + Ns;
% 
% end


%% generate data for new initial conditions
x_test = generate_data(modelFun, t_koop, x0, u_test);


%% plot results
if ~isnan(acc)

    if plot_results
        
        fig_modelcomp = plot_comparisons(x_test, x_koop, x0, t_koop);

    end

    if anim_results

        bernard = struct;
        bernard.xCenter = [0,0];
        bernard.radius = 0.25;
        bernard.distInfluence = 0.25;
        bernard.color = 'k';

        x_test_anim = x_test(:,end-(Ns-1):end);
        x_koop_anim = x_koop(:,end-(Ns-1):end);

        animate(bernard, x_koop_anim, tspan, world, xGoal(:,1), x_test_anim);

    end

end

function [x_n] = KoopFun(x, u, K, Q)
    Nx = length(x);
%     Nu = length(u);

    psi = observables(x, u, Q);
    psi_n = (K'*psi')';
    x_n = psi_n(1:Nx);
end











