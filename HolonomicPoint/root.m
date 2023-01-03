clean;

plot_results = 1;
anim_results = ~plot_results;

save_data = 0;


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
% load world environments (including obstacles)
load sphereworld_minimal


%% Model function
dt = 1;
modelFun = @(x, u) model(x, u, dt);


%% Initialize training data
N0 = 1;
Nx = 2;
Nu = Nx;

x0 = 10*rand(N0, Nx) - 5;

% simulation variables
T = 100;  tspan = 0:dt:T;
Nt = length(tspan);

% create list of inputs
A = 5;
u0 = A*rand(1,Nu) - A/2;
% uList = u0.*ones(Nt-1,Nu);                 % constant random input

% uList = u0 + (0.50*rand(Nt-1,Nu) - 0.25);  % input with noise
uList = A*[                                % sinusoidal input
    cos(linspace(0, 6*pi, Nt-1)'), -cos(linspace(0, 4*pi, Nt-1)')
];


%% Evaluate for the observation function
[~, meta] = observables(zeros(1,Nx), zeros(1,Nu), world);
Nk = meta.Nk(end);

observation = @(x, u) observables(x, u, world);
[K] = KoopmanAnalytical(world, meta);


%% initial observables
Psi0 = observation(x0, zeros(1,Nu));


%% generate data for new initial conditions
koop = @(x, u) KoopFun(x, u, K, world, meta);

PsiKoop = generate_data(koop, tspan, Psi0, uList, Nu);
xList = generate_data(modelFun, tspan, x0, uList, Nu);

PsiTest = NaN(Nt, Nk);
PsiTest(1,:) = observation(xList(1,:), zeros(1,Nu));
for i = 2:Nt
    PsiTest(i,:) = observation(xList(i,:), uList(i-1,:));
end

col = 1:meta.Nk;
PsiError = PsiTest(:,col)-PsiKoop(:,col) < 1e-3;
SumError = sum(PsiError, 'all');


%% plot results
if plot_results

    col = meta.xx;
    fig_comp = plot_comparisons(PsiTest(2:end,col), PsiKoop(2:end,col), Psi0(1,col), tspan(2:end), [], meta.labels(col));
    disp(meta.labels(col));

end

if anim_results

    bernard = struct;
    bernard.x = PsiKoop(:,meta.x);
    bernard.r = 0.25;
    bernard.color = 'k';

    x_test_anim = xList(:,meta.x);
    x_koop_anim = PsiKoop(:,meta.x);

    animate(world, bernard, [0,0], tspan, x_test_anim, x_koop_anim);

end


%% save data
if save_data
    save("./data/K_"+Nk+"x"+Nk, "K", "Nk", "dt", "meta", "Nw")
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

    Psi_n = observables(Psi(meta.x), u, world)*K;

end











