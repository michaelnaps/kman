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
dt = 0.1;
modelFun = @(x, u) model(x, u, dt);


%% Initialize training data
N0 = 1;
Nx = 2;
Nu = Nx;

x0 = 10*rand(Nx, N0) - 5;

% simulation variables
T = 10;  tspan = 0:dt:T;
Nt = length(tspan);

% create list of inputs
A = 5*rand - 2.5;
u0 = A*rand(Nu,1) - A/2;
% uList = u0.*ones(Nt-1,Nu);                 % constant random input

% uList = u0 + (0.50*rand(Nt-1,Nu) - 0.25);  % input with noise
uList = A*[                                % sinusoidal input
    cos(linspace(0, 6*pi, Nt-1)); -cos(linspace(0, 4*pi, Nt-1))
];


%% Evaluate for the observation function
[~, meta] = observables(zeros(Nx,1), zeros(Nu,1), world);
Nk = meta.Nk(end);

observation = @(x, u) observables(x, u, world);
K = KoopmanAnalytical(world, meta, dt);


%% initial observables
Psi0 = observation(x0, zeros(Nu,1));


%% generate data for new initial conditions
koop = @(x, u) KoopPropagate(x, u, K, world, meta);

PsiKoop = generate_data(koop, tspan, Psi0, uList, Nu);
xList = generate_data(modelFun, tspan, x0, uList, Nu);

PsiTest = NaN(Nk, Nt);
PsiTest(:,1) = observation(xList(:,1), zeros(Nu,1));
for i = 2:Nt
    PsiTest(:,i) = observation(xList(:,i), uList(:,i-1));
end

% col = 1:meta.Nk;
% PsiError = PsiTest(:,col)-PsiKoop(:,col) < 1e-3;
% SumError = sum(PsiError, 'all');


%% plot results
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
        row = meta.(fields{i});
        fig_comp = plot_comparisons(PsiTest(row,2:end), PsiKoop(row,2:end), Psi0(row,1), tspan(2:end), [], meta.labels(row), positions(i,:));
        disp(meta.labels(row));
    end

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
    save("../Data/K_"+Nk+"x"+Nk+"_analytical", "K", "Nk", "dt", "meta", "Nw")
end


%% local functions
function [Psi_n] = KoopPropagate(Psi, u, K, world, meta)

%     [dPsix, dPsiu] = observables_partial(Psi(meta.x), u, world);
%     Psi_n = Psi(meta.x)*dPsix*K + u*dPsiu*K;
%
%     disp("x")
%     disp((Psi(meta.x)*dPsix*K)')
% 
%     disp("u")
%     disp((u*dPsiu*K)')

    x = Psi(meta.x);

    xu = x*u';
    uu = u*u';

    Psi(meta.u) = u;
    Psi(meta.uu) = [uu(1), uu(2), uu(4)];
    Psi(meta.xu) = xu(:);

    Psi_n = K*Psi;

end