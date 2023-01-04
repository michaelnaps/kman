clean;

plot_results = 1;
anim_results = ~plot_results;

save_data = 1;

%% path environments
addpath ..
addpath ../Data
addpath ../DataFunctions
addpath ../KoopFunctions
addpath ../PlotFunctions
addpath ../SphereWorld

%% default plotting parameters
set(groot, 'DefaultLineLineWidth', 2);


%% initialize environment variables
load sphereworld_minimal


%% Model function
dt = 0.1;
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
uGenerate = NaN(Nt, N0*Nu);

k = 1;
for i = 1:N0
    uGenerate(:,k:k+Nu-1) = u0(i,:).*ones(Nt,Nu);
    k = k + Nu;
end

% generate model data
dataTrain = generate_data(modelFun, tspan, x0, uGenerate);
xTrain = stack_data(dataTrain, N0, Nx, Nt);
uTrain = stack_data(uGenerate, N0, Nu, Nt);


%% Evaluate for the observation function
[~, meta] = observables(zeros(1,Nx), zeros(1,Nx), world);
Nk = meta.Nk;

% use meta, ensure d is not factored into observable propagations
% depend = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1]';

observation = @(x, u) observables(x, u, world);
[K, acc, ind, err] = KoopmanWithControl(observation, xTrain, x0, uTrain);


%% test koopman operator on new data
N0 = 1;
Nx = 2;
Nu = Nx;

x0 = 10*rand(N0, Nx) - 5;

% simulation variables
T = 10;  tspan = 0:dt:T;
Nt = length(tspan);

% create list of inputs
A = 5*rand - 2.5;
uTest = A*[                                % sinusoidal input
    cos(linspace(0, 6*pi, Nt-1)'), -cos(linspace(0, 4*pi, Nt-1)')
];

% initial observables
Psi0 = observation(x0, zeros(1,Nu));


%% generate data for new initial conditions
koop = @(x, u) KoopFun(x, u, K, world, meta);

PsiKoop = generate_data(koop, tspan, Psi0, uTest, Nu);
xTest = generate_data(modelFun, tspan, x0, uTest, Nu);

PsiTest = NaN(Nt, Nk);
PsiTest(1,:) = observation(xTest(1,:), zeros(1,Nu));
for i = 2:Nt
    PsiTest(i,:) = observation(xTest(i,:), uTest(i-1,:));
end

col = meta.xx;
PsiError = abs(PsiTest(:,col)-PsiKoop(:,col)) < 1e-3;
SumError = sum(PsiError, 'all');


%% plot results
if ~isnan(acc)

    if plot_results
    
        fields = fieldnames(meta);
    
        def_size = [400, 250];
        positions = [
            1921, -70, def_size;
            2321, -70, def_size;
            1921, -420, def_size;
            2321, -413, def_size;
            1921, -755, def_size;
            2321, -755, def_size;
        ];
    
        for i = 1:length(fields)-3
            col = meta.(fields{i});
            fig_comp = plot_comparisons(PsiTest(2:end,col), PsiKoop(2:end,col), Psi0(1,col), tspan(2:end), [], meta.labels(col), positions(i,:));
            disp(meta.labels(col));
        end
    
    end

    if anim_results

        bernard = struct;
        bernard.x = PsiKoop(:,meta.x);
        bernard.r = 0.25;
        bernard.color = 'k';

        x_test_anim = xTest(:,meta.x);
        x_koop_anim = PsiKoop(:,meta.x);

        animate(world, bernard, [0,0], tspan, x_test_anim, x_koop_anim);

    end

end


%% save data
if save_data
    save("../Data/K_"+Nk+"x"+Nk+"_datadriven", "K", "Nk", "dt", "meta", "Nw")
end


%% local functions
function [Psi_n] = KoopFun(Psi, u, K, world, meta)

%     [dPsix, dPsiu] = observables_partial(Psi(meta.x), u, world);
%     Psi_n = Psi(meta.x)*dPsix*K + u*dPsiu*K;
%
%     disp("x")
%     disp((Psi(meta.x)*dPsix*K)')
% 
%     disp("u")
%     disp((u*dPsiu*K)')

    x = Psi(meta.x);

    xu = x'*u;
    uu = u'*u;

    Psi(meta.u) = u;
    Psi(meta.xu) = xu(:);
    Psi(meta.uu) = [uu(1), uu(2), uu(4)];

    Psi_n = Psi*K;

end











