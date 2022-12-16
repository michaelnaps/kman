clean;

plot_results = 1;
anim_results = ~plot_results;

save_data = 0;

%% path environments
addpath ./Data
addpath ./DataFunctions
addpath ./KoopFunctions
addpath ./PlotFunctions
addpath ./SphereWorld

%% default plotting parameters
set(groot, 'DefaultLineLineWidth', 2);


%% initialize environment variables
% load world environments (including obstacles)
load sphereworld_minimal


%% Model function
dt = 1;
modelFun = @(x, u) model(x, u, dt);


%% Initialize training data
Nrand = 50;
x0 = [
    20*rand(Nrand, 2) - 10;
    0, 0
];
[N0, Nx] = size(x0);
Nu = Nx;

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
[~, META] = observables(zeros(1,Nx), zeros(1,Nx), world);
Nk = META.Nk;

observation = @(x, u) observables(x, u, world);
[K, acc, ind, err] = KoopmanWithControl(observation, x_train, x0, u_train);


%% test koopman operator on new data
% time variables
TKoop = 20;
tKoop = (0:dt:TKoop)';
Nt = length(tKoop);

% introduce variance into the initial conditions
x0 = 20*rand(1,Nx) - 10;
[N0, Nx] = size(x0);

Psi0 = NaN(N0, Nk);

% create list of inputs
u0 = 5*rand(N0,Nu) - 2.5;
uTest = NaN(Nt,N0*Nu);

Nl = round(3/4*Nt);
Nz = Nt - Nl;

% create input matrices for time-frame
k = 1;
for i = 1:N0
    uTest(:,k:k+Nu-1) = [
        linspace(u0(i,1),0,Nl)', linspace(0,u0(i,2),Nl)';
        zeros(Nz, Nu);
    ];
    
    Psi0(i,:) = observation(x0(i,:), uTest(1,k:k+Nu-1));

    k = k + Nu;
end


%% generate data for new initial conditions
koop = @(x, u) KoopFun(x, u, K, world, META);

PsiKoop = generate_data(koop, tKoop, Psi0, uTest, Nu);
xTest = generate_data(modelFun, tKoop, x0, uTest, Nu);

PsiTest = NaN(size(PsiKoop(:,1:Nk)));
for i = 1:Nt
    PsiTest(i,:) = observation(xTest(i,1:Nx), uTest(i,1:Nu));
end

col = META.xx;
PsiError = PsiTest(:,col)-PsiKoop(:,col) < 1e-3;
SumError = sum(PsiError, 'all');


%% plot results
if ~isnan(acc)

    if plot_results

        col = META.x;
        fig_comp = plot_comparisons(PsiTest(2:end,col), PsiKoop(2:end,col), Psi0(1,col), tKoop(2:end));

    end

    if anim_results

        bernard = struct;
        bernard.x = PsiKoop(:,META.x);
        bernard.r = 0.25;
        bernard.color = 'k';

        x_test_anim = xTest(:,META.x);
        x_koop_anim = PsiKoop(:,META.x);

        animate(world, bernard, [0,0], tspan, x_test_anim, x_koop_anim);

    end

end


%% save data
if save_data
    save("./data/K_"+Nk+"x"+Nk, "K", "Nk", "dt", "META", "Nw")
end


%% local functions
function [Psi_n] = KoopFun(Psi, u, K, world, META)

    [dPsix, dPsiu] = observables_partial(Psi(META.x), u, world);
    Psi_n = Psi(META.x)*dPsix*K + u*dPsiu*K;

end











