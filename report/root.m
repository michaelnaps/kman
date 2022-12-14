%% clean workspace
clc;clear;
close all;


%% path environments
addpath ./Data
addpath ./DataFunctions
addpath ./KoopFunctions
addpath ./PlotFunctions
addpath ./SphereWorld

%% default plotting parameters
set(groot, 'DefaultLineLineWidth', 2);


%% initialize environment variables
% load world environments (including obstacles)
load sphereworld_minimal

% load pre-calculated Koopman operators
load K_24x24_datadriven
K_dd = K;

load K_24x24_analytical
K_an = K;


%% initialize plotting data
N0 = 1;
Nx = 2;
Nu = Nx;

% sim-time variables
T = 100;  tspan = (0:dt:T)';
Nt = length(tspan);

% initial state
x0 = 10*rand(N0,Nx) - 5;

% input list
A = 2.5;                                   % maximum input
u0 = zeros(1,Nu);                          % initial input 1
% u0 = A*rand(1,Nu) - A/2;                   % initial input 2
% uList = u0.*ones(Nt-1,Nu);                 % constant input
% uList = u0 + (0.50*rand(Nt-1,Nu) - 0.25);  % input with noise
uList = A*[                                % sinusoidal input
    cos(linspace(0, 6*pi, Nt-1)'), -cos(linspace(0, 4*pi, Nt-1)')
];


%% grab meta-data variable
[Psi0, META] = observables(x0, u0, world);
observationFun = @(x,u) observables(x, u, world);


%% propagation functions
ModelPropagate = @(x,u) model(x, u, dt);
KoopAnalytical = @(Psi,u) KoopPropagate(Psi, u, K_an, world, META);
KoopDataDriven = @(Psi,u) KoopPropagate(Psi, u, K_dd, world, META);


%% generate simulation data
PsiList_Analytical = generate_data(KoopAnalytical, tspan, Psi0, uList, Nu);
PsiList_DataDriven = generate_data(KoopDataDriven, tspan, Psi0, uList, Nu);

xList_Model = generate_data(ModelPropagate, tspan, x0, uList);

PsiList_Model = NaN(Nt, META.Nk);
PsiList_Model(1,:) = observationFun(xList_Model(1,:), u0);
for i = 2:Nt
    PsiList_Model(i,:) = observationFun(xList_Model(i,:), uList(i-1,:));
end


%% plot comparison results
% line colors (for consistency)
ModelColor = [0 0.4470 0.7410];
AnalyticalColor = [0.8500 0.3250 0.0980];
DataDrivenColor = [0.4660 0.6740 0.1880];

% results for x propagation (position)
idx = META.x;
figure('Position', [1921,397,1080,592])
subplot(2,2,1);
    hold on
    plot(tspan, PsiList_Model(:,idx(1)), 'LineWidth', 2.5, 'Color', ModelColor);
    plot(tspan, PsiList_Analytical(:,idx(1)), '--', 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(1)), ':', 'Color', DataDrivenColor);
    hold off
subplot(2,2,2);
    hold on
    plot(tspan, PsiList_Model(:,idx(2)), 'Color', ModelColor);
    plot(tspan, PsiList_Analytical(:,idx(2)), '--', 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(2)), ':', 'Color', DataDrivenColor);
    legend("Model", "Analytical", "Data-Driven")
    hold off
subplot(2,2,3);
    hold on
    plot(tspan, PsiList_Analytical(:,idx(1))-PsiList_Model(:,idx(1)), '--', 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(1))-PsiList_Model(:,idx(1)), ':', 'Color', DataDrivenColor);
    hold off
subplot(2,2,4);
    hold on
    plot(tspan, PsiList_Analytical(:,idx(2))-PsiList_Model(:,idx(2)), '--', 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(2))-PsiList_Model(:,idx(2)), ':', 'Color', DataDrivenColor);
    legend("Analytical", "Data-Driven")
    hold off

% results for x'x propagation
idx = META.xx;
figure('Position', [1921,397,1080,592])
subplot(2,4,1);
    hold on
    plot(tspan, PsiList_Model(:,idx(1)), 'LineWidth', 2.5, 'Color', ModelColor);
    plot(tspan, PsiList_Analytical(:,idx(1)), '--', 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(1)), ':', 'Color', DataDrivenColor);
    hold off
subplot(2,4,2);
    hold on
    plot(tspan, PsiList_Model(:,idx(2)), 'Color', ModelColor);
    plot(tspan, PsiList_Analytical(:,idx(2)), '--', 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(2)), ':', 'Color', DataDrivenColor);
    legend("Model", "Analytical", "Data-Driven")
    hold off
subplot(2,4,3);
    hold on
    plot(tspan, PsiList_Model(:,idx(3)), 'LineWidth', 2.5, 'Color', ModelColor);
    plot(tspan, PsiList_Analytical(:,idx(3)), '--', 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(3)), ':', 'Color', DataDrivenColor);
    hold off
subplot(2,4,4);
    hold on
    plot(tspan, PsiList_Model(:,idx(4)), 'Color', ModelColor);
    plot(tspan, PsiList_Analytical(:,idx(4)), '--', 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(4)), ':', 'Color', DataDrivenColor);
    legend("Model", "Analytical", "Data-Driven")
    hold off

subplot(2,4,5);
    hold on
    plot(tspan, PsiList_Analytical(:,idx(1))-PsiList_Model(:,idx(1)), '--', 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(1))-PsiList_Model(:,idx(1)), ':', 'Color', DataDrivenColor);
    hold off
subplot(2,4,6);
    hold on
    plot(tspan, PsiList_Analytical(:,idx(2))-PsiList_Model(:,idx(2)), '--', 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(2))-PsiList_Model(:,idx(2)), ':', 'Color', DataDrivenColor);
    legend("Analytical", "Data-Driven")
    hold off
subplot(2,4,7);
    hold on
    plot(tspan, PsiList_Analytical(:,idx(3))-PsiList_Model(:,idx(3)), '--', 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(3))-PsiList_Model(:,idx(3)), ':', 'Color', DataDrivenColor);
    hold off
subplot(2,4,8);
    hold on
    plot(tspan, PsiList_Analytical(:,idx(4))-PsiList_Model(:,idx(4)), '--', 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(4))-PsiList_Model(:,idx(4)), ':', 'Color', DataDrivenColor);
    legend("Analytical", "Data-Driven")
    hold off


%% local functions
function [Psi_n] = KoopPropagate(Psi, u, K, world, META)

    [dPsix, dPsiu] = observables_partial(Psi(META.x), u, world);
    Psi_n = Psi(META.x)*dPsix*K + u*dPsiu*K;

end