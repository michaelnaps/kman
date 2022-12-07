clean;

plot_results = 1;
anim_results = ~plot_results;

save_data = 0;

addpath ../.
addpath ../../.
addpath ../sphereworld;

load sphereworld_minimal;


%% Model function
dt = 0.01;
modelFun = @(x, u) model(x, u, dt);


%% Initialize training data
Nrand = 30;
x0 = [
    20*rand(Nrand, 2), 10*rand(Nrand, 2) - 5;
    0, 0, 20, 10
];
[N0, Nx] = size(x0);
Nu = round(Nx/2);

% simulation variables
T = 10;
tspan = 0:dt:T;
Nt = length(tspan);

% create list of inputs
u0 = 20*rand(N0, Nu) - 10;
u_generate = NaN(Nt, N0*Nu);

k = 1;
for i = 1:N0
    u_generate(:,k:k+Nu-1) = [
        linspace(u0(i,1),0,Nt)', linspace(u0(i,2),0,Nt)'
    ];
    k = k + Nu;
end

% generate model data
data_train = generate_data(modelFun, tspan, x0, u_generate);
x_train = stack_data(data_train, N0, Nx, Nt);
u_train = stack_data(u_generate, N0, Nu, Nt);


%% Evaluate for the observation function
Q = 1;
Nk = (Nx + Nw)*Q + Nu;

observation = @(x, u) observables(x, u, Q, world);

[K, acc, ind, err] = KoopmanWithControl(observation, x_train, x0, u_train);
fprintf("L-2 norm: %.3f\n\n", acc)


%% test koopman operator on new data
% time variables
T_koop = 20;
t_koop = (0:dt:T_koop)';
Nt = length(t_koop);

% introduce variance into the initial conditions
x0 = x0(N0-4:end,:);
[N0, Nx] = size(x0);
x0 = x0 + [(rand(N0-1, Nx) - 0.5); 0, 0, 0, 0];
Psi0 = NaN(N0,Nk);

% create list of inputs
u0 = 20*rand(N0,Nu) - 10;
u_test = NaN(Nt,N0*Nu);

Nl = round(Nt/4);
Nz = Nt - Nl;

% create input matrices for time-frame
k = 1;
for i = 1:N0
    u_test(:,k:k+Nu-1) = [
        linspace(u0(i,1),0,Nl)', linspace(0,u0(i,2),Nl)';
        zeros(Nz, Nu);
    ];
    
    Psi0(i,:) = observation(x0(i,:), [0,0]);

    k = k + Nu;
end


%% generate data for new initial conditions
koop = @(x, u) KoopFun(x, u, K, Q);

Psi_koop = generate_data(koop, t_koop, Psi0, u_test, Nu);
x_test = generate_data(modelFun, t_koop, x0, u_test, Nu);


%% obstacle distance comparison
obs_koop = Psi_koop(:,Q*Nx+1:Q*Nx+Nw);
obs_test = NaN(Nt,Nw);

for i = 1:Nt
     psi_temp = observation(x_test(i,1:Nx), [0,0]);
     obs_test(i,:) = psi_temp(Q*Nx+1:Q*Nx+Nw);
end


%% plot results
if ~isnan(acc)

    if plot_results
        
%         fig_modelcomp = plot_comparisons(x_test, Psi_koop, x0, t_koop, Psi0);
        fig_obscomp   = plot_comparisons(obs_test, obs_koop, obs_test(1,:), t_koop);

    end

    if anim_results

        bernard = struct;
        bernard.xCenter = [0,0];
        bernard.radius = 0.25;
        bernard.distInfluence = 0.25;
        bernard.color = 'k';

        x_test_anim = x_test(:,end-(Nx-1):end);
        x_koop_anim = Psi_koop(:,end-(Nx-1):end);

        animate(bernard, x_koop_anim, tspan, world, xGoal(:,1), x_test_anim);

    end

end


%% save data
if save_data
    save("./data/K_"+Nk+"x"+Nk, "K", "Nk", "dt", "Q", "acc", "ind", "Nw")
end


%% local functions
function [Psi_n] = KoopFun(Psi, u, K, Q)
    Nx = 4;
    Nu = 2;
    Nw = 4;
    Nk = (Nx + Nw)*Q + Nu;

    dKx = diag([ones(1,Nk-Nu), zeros(1,Nu)]);
    dKu = diag([zeros(1,Nk-Nu), ones(1,Nu)]);

    uPsi = [zeros(1,Nk-Nu), u];

    Psi_n = Psi*dKx*K + uPsi*dKu*K;
end