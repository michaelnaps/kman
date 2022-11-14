clean;

plot_results = 1;
anim_results = ~plot_results;

addpath ../.
addpath ./sphereworld;

load sphereworld;


%% Model function
dt = 0.01;
modelFun = @(x, u) model(x, u, dt);
inputFun = @(x) rand(1,2);


%% Initialize training data
Nrand = 20;
x0 = [
    xStart', zeros(size(xStart')), ;
    10*rand(Nrand, 2), 4*rand(Nrand, 2) - 2;
];

Nu = length(inputFun(x0(1,:)));
[Nx, Ns] = size(x0);

% simulation variables
T = 10;
tspan = 0:dt:T;
Nt = length(tspan);

% generate model data
data_train = generate_data(modelFun, tspan, x0, inputFun);
x_train = stack_data(data_train, Nx, Ns+Nu, Nt);


%% Evaluate for the observation function
Q = 1;
Nk = (Ns+Nu)*Q^(Ns+Nu);

observation = @(x) observables(x, Q);

[K, acc, ind, err] = koopman(observation, x_train, x0, 2);
fprintf("L-2 norm: %.3s\n\n", acc)


%% test koopman operator on new data
% time variables
T_Koop = T;
t_Koop = (0:dt:T_Koop)';
Nt = length(t_Koop);

% introduce variance into the initial conditions
x0 = x0(Nx-10:end,:);
[Nx, Ns] = size(x0);
x0 = x0 + (rand(Nx, Ns) - 0.5);

psi0 = NaN(Nx, Nk);
for i = 1:Nx

    psi0(i,:) = observation(x0(i,:));

end

KoopFun = @(psi) (K'*psi')';
x_Koop = generate_data(KoopFun, t_Koop, psi0);

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

%     if anim_results
% 
%         names = ["", "Model", "", "Koopman"];
%         animate(t_Koop, x_modl, -x_Koop, names, 50, 2);
% 
%     end

end






















