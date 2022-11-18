clean;

plot_results = 1;
anim_results = ~plot_results;

addpath ../.
addpath ../../.
addpath ../sphereworld;

load sphereworld;


%% Model function
dt = 0.01;
modelFun = @(x, u) model(x, [0,0], dt);


%% Initialize training data
Nrand = 20;
x0 = [
    xStart', zeros(size(xStart'));
    20*rand(Nrand, 2), 10*rand(Nrand, 2) - 5;
    0, 0, 20, 10
];
[Nx, Ns] = size(x0);

% simulation variables
T = 10;
tspan = 0:dt:T;
Nt = length(tspan);

% generate model data
data_train = generate_data(modelFun, tspan, x0);
x_train = stack_data(data_train, Nx, Ns, Nt);


%% Evaluate for the observation function
Q = 1;
Nk = Ns*Q;

observation = @(x, u) observables(x, [0,0], Q);

[K, acc, ind, err] = koopman(observation, x_train, x0);
fprintf("L-2 norm: %.3s\n\n", acc)


%% test koopman operator on new data
% time variables
T_Koop = T;
t_Koop = (0:dt:T_Koop)';
Nt = length(t_Koop);

% introduce variance into the initial conditions
x0 = x0(Nx-4:end,:);
[Nx, Ns] = size(x0);
x0 = x0 + [(rand(Nx-1, Ns) - 0.5); 0, 0, 0, 0];

psi0 = NaN(Nx, Nk);
for i = 1:Nx

    psi0(i,:) = observation(x0(i,:));

end

KoopFun = @(psi) (K'*psi')';
data_Koop = generate_data(KoopFun, t_Koop, psi0);

% delete unwanted elements from the observation space
k = 1;  j = 1;
x_Koop = NaN(Nt,Ns*Nx);
for i = 1:Nx

    x_Koop(:,j:j+Ns-1) = data_Koop(:,k:k+Ns-1);

    k = k + Nk;
    j = j + Ns;

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

        x_test_anim = x_test(:,end-(Ns-1):end);
        x_Koop_anim = x_Koop(:,end-(Ns-1):end);

        animate(bernard, x_Koop_anim, tspan, world, xGoal(:,1), x_test_anim);

    end

end