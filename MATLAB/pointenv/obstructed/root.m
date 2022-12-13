clean;

plot_results = 1;
anim_results = ~plot_results;

save_data = 1;

addpath ../.
addpath ../../.
addpath ../sphereworld;

load sphereworld_minimal;
Nw = length(world);


%% Model function
dt = 1;
modelFun = @(x, u) model(x, u, dt);


%% Initialize training data
N0 = 1;
Nx = 2;
Nu = Nx;

x0 = 20*rand(N0, Nx) - 10;

% simulation variables
T = 100;  tspan = 0:dt:T;
Nt = length(tspan);

% create list of inputs
u0 = 20*rand(N0, Nu) - 10;
uList = u0.*ones(Nt,Nu);

% generate model data
data_train = generate_data(modelFun, tspan, x0, uList);

% stack data
xTrain = stack_data(data_train, N0, Nx, Nt);
uList = stack_data(uList, N0, Nu, Nt);


%% Evaluate for the observation function
[~, META] = observables(zeros(1,Nx), zeros(1,Nx), world);
Nk = META.Nk(end);

observation = @(x, u) observables(x, u, world);
% [K, acc, ind, err] = KoopmanWithControl(observation, xTrain, x0, uTrain);
[K] = KoopmanAnalytical(world, META);  acc = 1;


%% initial observables
Psi0 = NaN(N0,Nk);
for i = 1:N0
    Psi0(i,:) = observation(x0(i,:), zeros(1,Nu));
end


%% generate data for new initial conditions
koop = @(x, u) KoopFun(x, u, K, world, META);

PsiKoop = generate_data(koop, tspan, Psi0, uList, Nu);
xTest = generate_data(modelFun, tspan, x0, uList, Nu);

PsiTest = NaN(Nt, Nk);
PsiTest(1,:) = observation(xTest(1,META.x), [0,0]);
for i = 2:Nt
    PsiTest(i,:) = observation(xTest(i,1:Nx), uList(i-1,1:Nu));
end

col = META.xx;
PsiError = PsiTest(:,col)-PsiKoop(:,col) < 1e-3;
SumError = sum(PsiError, 'all');


%% plot results
if ~isnan(acc)

    if plot_results

        col = META.x;
        fig_comp = plot_comparisons(PsiTest(:,col), PsiKoop(:,col), Psi0(1,col), tspan);

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
    save("./data/K_"+Nk+"x"+Nk, "K", "Nk", "dt", "acc", "META", "Nw")
end


%% local functions
function [Psi_n] = KoopFun(Psi, u, K, world, META)

    [dPsix, dPsiu] = observables_partial(Psi(META.x), u, world);
    Psi_n = Psi(META.x)*dPsix*K + u*dPsiu*K;

end











