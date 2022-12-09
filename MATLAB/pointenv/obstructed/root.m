clean;

plot_results = 1;
anim_results = ~plot_results;

save_data = 0;

addpath ../.
addpath ../../.
addpath ../sphereworld;

load sphereworld_nowall;


%% Model function
dt = 0.01;
modelFun = @(x, u) model(x, u, dt);


%% Initialize training data
Nrand = 10;
x0 = [
    20*rand(Nrand, 2)-10;
    0, 0
];
[N0, Nx] = size(x0);
Nu = Nx;

% simulation variables
T = 10;
tspan = 0:dt:T;
Nt = length(tspan);

% create list of inputs
u0 = 5*rand(N0, Nu) - 2.5;
uGenerate = NaN(Nt, N0*Nu);

k = 1;
for i = 1:N0
    uGenerate(:,k:k+Nu-1) = [
        linspace(u0(i,1),0,Nt)', linspace(u0(i,2),0,Nt)'
    ];
    k = k + Nu;
end

% generate model data
dataTrain = generate_data(modelFun, tspan, x0, uGenerate);
xTrain = stack_data(dataTrain, N0, Nx, Nt);
uTrain = stack_data(uGenerate, N0, Nu, Nt);


%% Evaluate for the observation function
Q = 2;
observation = @(x, u) observables(x, u, Q, world);
[~, Nk, INDEX] = observation(x0(1,:), u0(1,:));

[K, acc, ind, err] = KoopmanWithControl(observation, xTrain, x0, uTrain);
fprintf("L-2 norm: %.3f\n\n", acc)


%% test koopman operator on new data
% time variables
T_koop = 20;
t_koop = (0:dt:T_koop)';
Nt = length(t_koop);

% introduce variance into the initial conditions
x0 = x0(N0-4:end,:);
[N0, Nx] = size(x0);
x0 = x0 + [(rand(N0-1, Nx) - 0.5); zeros(1,Nx)];
Psi0 = NaN(N0,Nk);

% create list of inputs
u0 = 5*rand(N0,Nu) - 2.5;
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
koop = @(Psi, u) KoopFun(Psi, u, K, Q, INDEX);
koop2 = @(Psi, u) KoopFun2(Psi, u, K, INDEX);

PsiKoop = generate_data(koop, t_koop, Psi0, u_test, Nu);
xTest = generate_data(modelFun, t_koop, x0, u_test, Nu);


%% obstacle distance comparison
obs_koop = PsiKoop(:,Q*Nx+1:Q*Nx+Nw);
obs_test = NaN(Nt,Nw);

for i = 1:Nt
     psi_temp = observation(xTest(i,1:Nx), [0,0]);
     obs_test(i,:) = psi_temp(Q*Nx+1:Q*Nx+Nw);
end


%% plot results
if ~isnan(acc)

    if plot_results
        
        fig_modelcomp = plot_comparisons(xTest, PsiKoop, x0, t_koop, Psi0);
        fig_obscomp   = plot_comparisons(obs_test, obs_koop, obs_test(1,:), t_koop);

    end

    if anim_results

        bernard = struct;
        bernard.x = PsiKoop(:,INDEX.x1);
        bernard.r = 0.25;
        bernard.color = 'k';

        x_test_anim = xTest(:,1:Nx);
        x_koop_anim = PsiKoop(:,1:Nx);

        animate(bernard, x_test_anim, tspan, world, [0,0], x_koop_anim);

    end

end


%% save data
if save_data
    save("./data/K_"+Nk+"x"+Nk, "K", "Nk", "dt", "Q", "acc", "ind", "Nw")
end


%% local functions
function [Psi_n] = KoopFun(Psi, u, K, Q, INDEX)
    Nx = length(INDEX.x1);
    Nw = length(INDEX.d);
    Nu = length(INDEX.u);
    Nxu = length(INDEX.xu);
    No = Nw*length(INDEX.o1);
    Nk = Q*Nx + Nw + Nu + Nxu + No + 1;

    dKx = diag([
        1, 1,...
        1, 1,...
        1, 1, 1,...
        0, 0,...
        1, 1,...
        0, 0,...
        1, 1,...
        0, 0,...
        0,...
        1, 1,...
        1, 1,...
        1, 1,...
    ]);

    dKu = diag([
        0, 0,...
        0, 0,...
        1, 1, 1,...
        1, 1,...
        1, 1,...
        1, 1,...
        1, 1,...
        1, 1,...
        0,...
        0, 0,...
        0, 0,...
        0, 0,...
    ]);

    uPsi = zeros(1,Nk);
    uPsi(INDEX.u) = u;

    Psi_n = Psi*dKx*K + uPsi*dKu*K;
end

function [Psi_n] = KoopFun2(Psi, u, K, INDEX)
    Psi(INDEX.u) = u;
    Psi_n = Psi*K;
end









