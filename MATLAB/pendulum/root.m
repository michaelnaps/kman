clean;

anim_results = 0;
plot_results = ~anim_results;


%% initialize variables
th0 = [
    pi/2-0.01, 0;
    pi/2+0.1, 0;
    pi/2, -1;
    pi/2, 1;
    3*pi/2, 0;
    3*pi/2, 0.1;
    3*pi/2, -0.1;
    0, 0;
    0, 0.1;
    0, -0.1;
    pi, 0;
    pi, 0.1;
    pi, -0.1
    pi/2, 0;
];

[Nth, Ns] = size(th0);

dt = 0.001; T = 10;
tspan = (0:dt:T)';
M = length(tspan);

modelFun = @(x) model(x, dt);

data_modl = generate_data(modelFun, tspan, th0);

Ncol = numel(data_modl)/Ns;
th_modl = [
    reshape(data_modl(:,1:Ns:Ns*Nth-1), Ncol, 1), reshape(data_modl(:,Ns:Ns:Ns*Nth), Ncol, 1)
];


%% evaluate for the observation function
Q = 1;
Nk = 2*Q^2 + 2;

observation = @(x) observables(x, Q);
[K, acc, ind, err] = koopman(observation, th_modl, th0);
fprintf("L-2 norm: %.3s\n\n", acc)


%% test koopman operator on new data
% addpath ./test
% load WS_pend.mat;

% time variables
T_Koop = 10;
t_Koop = (0:dt:T_Koop)';
Nt = round(T_Koop/dt)+1;

% redeclare functions (for reading old data)
modelFun = @(x) model(x, dt);
observation = @(x) observables(x, Q);

% introduce variance into the initial conditions
th0 = th0(round(Nth/2):end,:);
[Nth, Ns] = size(th0);
th0 = th0 + 0.5*[rand(Nth-1, Ns); 0, 0];

psi0 = NaN(Nth, Nk);
for i = 1:Nth
    psi0(i,:) = observation(th0(i,:));
end

KoopFun = @(psi) (K'*psi')';
th_Koop = generate_data(KoopFun, t_Koop, psi0);

k = 1;
for i = 1:Nth

    th_Koop(:,k+2:k+Nk-1) = [];
    k = k + Ns;

end


%% calculate reference values
th_modl = generate_data(modelFun, t_Koop, th0);


%% plot results
if ~isnan(acc)

    if plot_results
        
        fig_modelcomp = plot_comparisons(th_modl, th_Koop, th0, t_Koop);

    end

    if anim_results

        names = ["", "Model", "", "Koopman"];
        animate(t_Koop, th_modl, -th_Koop, names, 50, 2);

    end

end