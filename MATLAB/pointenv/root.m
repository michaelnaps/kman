clean;

plot_results = 1;
anim_results = ~plot_results;

addpath ./kman;
addpath ./sphereworld;

load sphereworld;


%% Model function
dt = 1;
modelFun = @(x) model(x, [0,0], dt);


%% Initialize training data
x0 = [
    xStart', zeros(size(xStart'));
    10*rand(15, 2), 4*rand(15, 2) - 2;
];
[Nx, Ns] = size(x0);

% simulation variables
T = 10;
tspan = 0:dt:T;
Nt = length(tspan);

% generate model data
data_train = generate_data(modelFun, tspan, x0);
x_train = stack_data(data_train, Nx, Ns, Nt);

u = [0, 0];


%% Evaluate for the observation function
Q = 1;
Nu = length(u);
Nk = (Ns+Nu)*Q^(Ns+Nu);

observation = @(x) observables(x, u, Q);

[K, acc, ind, err] = koopman(observation, x_train, x0);
fprintf("L-2 norm: %.3s\n\n", acc)


%% test koopman operator on new data
% time variables
T_Koop = T;
t_Koop = (0:dt:T_Koop)';
Nt = length(t_Koop);

% redeclare functions (for reading old data)
modelFun = @(x) model(x, u, dt);
observation = @(x) observables(x, u, Q);

% introduce variance into the initial conditions
x0 = x0(Nx-4:end,:);
[Nx, Ns] = size(x0);
x0 = x0 + 0.5*rand(Nx, Ns);

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
        
        fig_modelcomp = plot_comparisons(data_train, x_Koop, x0, t_Koop);

    end

%     if anim_results
% 
%         names = ["", "Model", "", "Koopman"];
%         animate(t_Koop, x_modl, -x_Koop, names, 50, 2);
% 
%     end

end






















