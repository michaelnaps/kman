clean;

plot_results = 1;
anim_results = ~plot_results;

save_data = 0;

addpath ../.
addpath ../../.
addpath ../sphereworld;

load sphereworld_minimal;
Nw = length(world);


%% Model function
dt = 0.01;
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
Q = 5;
[~, Nk, META] = observables(zeros(1,Nx), zeros(1,Nx));

observation = @(x, u) observables(x, u);

[K, acc, ind, err] = KoopmanWithControl(observation, x_train, x0, u_train);
fprintf("L-2 norm: %.3f\n\n", acc)


%% test koopman operator on new data
% time variables
TKoop = 20;
tKoop = (0:dt:TKoop)';
Nt = length(tKoop);

% introduce variance into the initial conditions
x0 = x0(N0-4:end,:);
[N0, Nx] = size(x0);
x0 = x0 + [(rand(N0-1, Nx) - 0.5); zeros(1,Nx)];
Psi0 = NaN(N0,Nk);

% create list of inputs
u0 = 5*rand(N0,Nu) - 2.5;
uTest = NaN(Nt,N0*Nu);

Nl = round(Nt/4);
Nz = Nt - Nl;

% create input matrices for time-frame
k = 1;
for i = 1:N0
    uTest(:,k:k+Nu-1) = [
        linspace(u0(i,1),0,Nt)', linspace(0,u0(i,2),Nt)';
    ];
    
    Psi0(i,:) = observation(x0(i,:), uTest(1,k:k+Nu-1));

    k = k + Nu;
end


%% generate data for new initial conditions
koop = @(x, u) KoopFun(x, u, K, META);

PsiKoop = generate_data(koop, tKoop, Psi0, uTest, Nu);
xTest = generate_data(modelFun, tKoop, x0, uTest, Nu);

PsiTest = NaN(size(PsiKoop(:,1:Nk)));
for i = 1:Nt
    PsiTest(i,:) = observation(xTest(i,1:Nx), uTest(i,1:Nu));
end


%% plot results
if ~isnan(acc)

    if plot_results

        col = META.xu;
        fig_comp = plot_comparisons(PsiTest(:,col), PsiKoop(:,col)/2, Psi0(1,col), tKoop);

    end

    if anim_results

        bernard = struct;
        bernard.x = PsiKoop(:,META.x1);
        bernard.r = 0.25;
        bernard.color = 'k';

        x_test_anim = xTest(:,META.x1);
        x_koop_anim = PsiKoop(:,META.x1);

        animate(world, bernard, [0,0], tspan, x_test_anim, x_koop_anim);

    end

end


%% save data
if save_data
    save("./data/K_"+Nk+"x"+Nk, "K", "Nk", "dt", "acc", "META", "Nw")
end


%% local functions
function [Psi_n] = KoopFun(Psi, u, K, META)

    [dKx, dKu] = observables_partial(Psi(META.x), u);

%     size(K)
%     size(dKx)
%     size(dKu)
%     size(Psi)

    Psi_n = Psi(META.x)*dKx*K + u*dKu*K;
    
end











