clean;

plot_results = 1;
anim_results = ~plot_results;

save_data = 1;

addpath ../.
addpath ../../.
addpath ../sphereworld;

load sphereworld;
Nw = length(world);


%% Model function
dt = 0.01;
modelFun = @(x, u) model(x, u, dt);


%% Initialize training data
Nrand = 50;
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
Nk = Ns*Q + Nw + Nu;

observation = @(x, u) observables(x, u, Q, world);

[K, acc, ind, err] = KoopmanWithControl(observation, x_train, x0, u_train);
fprintf("L-2 norm: %.3f\n\n", acc)


%% test koopman operator on new data
% time variables
T_koop = 20;
t_koop = (0:dt:T_koop)';
Nt = length(t_koop);

% introduce variance into the initial conditions
x0 = x0(Nx-4:end,:);
[Nx, Ns] = size(x0);
x0 = x0 + [(rand(Nx-1, Ns) - 0.5); 0, 0, 0, 0];
Psi0 = NaN(Nx,Nk);

% create list of inputs
u0 = 20*rand(Nx,Nu) - 10;
u_test = NaN(Nt,Nx*Nu);
Nl = round(Nt/4);
N0 = Nt - Nl;

% create input matrices for time-frame
k = 1;
for i = 1:Nx
    u_test(:,k:k+Nu-1) = [
        linspace(u0(i,1),0,Nl)', linspace(0,u0(i,2),Nl)';
        zeros(N0, Nu);
    ];
    
    Psi0(i,:) = observation(x0(i,:), [0,0]);

    k = k + Nu;
end


%% generate data for new initial conditions
koop = @(x, u) KoopFun(x, u, K, Q, world);

Psi_koop = generate_data(koop, t_koop, Psi0, u_test, Nu);
x_test = generate_data(modelFun, t_koop, x0, u_test, Nu);


%% plot results
if ~isnan(acc)

    if plot_results
        
        fig_modelcomp = plot_comparisons(x_test, Psi_koop, x0, t_koop, Psi0);

    end

    if anim_results

        bernard = struct;
        bernard.xCenter = [0,0];
        bernard.radius = 0.25;
        bernard.distInfluence = 0.25;
        bernard.color = 'k';

        x_test_anim = x_test(:,end-(Ns-1):end);
        x_koop_anim = Psi_koop(:,end-(Ns-1):end);

        animate(bernard, x_koop_anim, tspan, world, xGoal(:,1), x_test_anim);

    end

end


%% save data
if save_data
    save("./data/K_"+Nk+"x"+Nk, "K", "Nk", "dt", "Q", "acc", "ind", "Nw")
end


%% local functions
function [Psi_n] = KoopFun(Psi, u, K, Q, world)
    Nx = 4;
    Nu = 2;
    Nw = length(world);
    Nk = Q*Nx + Nw + Nu;

    dKx = diag([ones(1,Nk-Nu), zeros(1,Nu)]);
    dKu = diag([zeros(1,Nk-Nu), ones(1,Nu)]);

    uPsi = [zeros(1,Nk-Nu), u];

    Psi_n = Psi*dKx*K + uPsi*dKu*K;
end