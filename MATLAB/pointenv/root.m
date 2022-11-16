clean;

plot_results = 1;
anim_results = ~plot_results;

addpath ../.
addpath ./sphereworld;

load sphereworld;


%% Model function
dt = 0.001;
inputFun = @(x) [0,0];
modelFun = @(x) model(x, dt, inputFun);


%% Initialize training data
Nrand = 2;
x0 = [
    xStart', zeros(size(xStart')), zeros(size(xStart'));
    10*rand(Nrand, 2), 4*rand(Nrand, 2) - 2, 2*rand(Nrand, 2) - 1;
];

Nu = length(inputFun(x0(1,:)));
[Nx, Ns] = size(x0);

% simulation variables
T = 20;
tspan = 0:dt:T;
Nt = length(tspan);

% generate model data
data_train = generate_data(modelFun, tspan, x0);
x_train = stack_data(data_train, Nx, Ns, Nt);


%% Evaluate for the observation function
Q = 2;
Nk = (Ns-2)*Q^(Ns-2) + 2;

observation = @(x) observables(x, inputFun, Q);

[K, acc, ind, err] = koopman(observation, x_train, x0, 2);
fprintf("L-2 norm: %.3s\n\n", acc)


%% test koopman operator on new data
% time variables
T_Koop = T;
t_Koop = (0:dt:T_Koop)';
Nt = length(t_Koop);

% introduce variance into the initial conditions
x0 = x0(Nx-4:end,:);
[Nx, Ns] = size(x0);
x0 = x0 + (rand(Nx, Ns) - 0.5);

psi0 = NaN(Nx, Nk);
for i = 1:Nx

    psi0(i,:) = observation(x0(i,:));

end

KoopFun = @(psi) (K'*psi')';
x_Koop = generate_data(KoopFun, t_Koop, psi0);

% delete unwanted elements from the observation space
k = 1;
for i = 1:Nx

    x_Koop(:,k+Ns:k+Nk-1) = [];
    k = k + Ns;

end


%% generate data for new initial conditions
x_test = generate_data(modelFun, t_Koop, x0);


%% plot results
if ~isnan(acc)

    if plot_results
        
        fig_modelcomp = plot_comparisons(x_test, x_Koop, x0, t_Koop);

    end

    if anim_results

        bernard = struct;
        bernard.xCenter = [0,0];
        bernard.radius = 0.25;
        bernard.distInfluence = 0.25;
        bernard.color = 'k';

        animate(bernard, x_Koop, tspan, world, [0;0]);

    end

end






















