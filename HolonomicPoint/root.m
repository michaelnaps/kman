%% clean workspace
clc;clear;
close all;

save_figures = 0;
figure_path = "/home/michaelnaps/bu_research/literature/koopman_collision_avoidance/figures/";


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

% load pre-calculated Koopman operators
load K_24x24_datadriven
K_dd = K;

% load K_24x24_analytical
K_an = KoopmanAnalytical(world, META);


%% initialize plotting data
N0 = 1;
Nx = 2;
Nu = Nx;

% sim-time variables
T = 100;  tspan = (0:dt:T)';
Nt = length(tspan);

% initial state
x0 = [0, 0];
% x0 = 10*rand(N0,Nx) - 5;

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


%% plotting parameters
tspan = tspan/10;

ModelColor = [0 0.4470 0.7410];
ModelLineWidth = 2.5;

ReferenceColor = ModelColor;
ReferenceMarker = "-.";

AnalyticalColor = [0.8500 0.3250 0.0980];
AnalyticalMarker = "--";

DataDrivenColor = [0.4660 0.6740 0.1880];
DataDrivenMarker = ":";


%% input trajectory over time
u_fig = figure('Position', [2425,397,459,193]);
hold on
plot(tspan(1:end-1), uList(:,1), "Color",  [255,195,0]/255);
plot(tspan(1:end-1), uList(:,2), "Color", [255,87,51]/255);
title("u")
ylabel("Velocity [m/s]");
xlabel("Time [s]")
legend("u_x", "u_y")
hold off

if save_figures
    figure_name = "u_propagation.png";
    exportgraphics(u_fig, figure_path + figure_name, 'Resolution', 600);
end


%% results for x propagation (position)
idx = META.x;
x_fig = figure('Position', [1921,397,459,320]);
% sgtitle("Propagation of x")
subplot(2,2,1);
    hold on
    plot(tspan, PsiList_Model(:,idx(1)), 'LineWidth', ModelLineWidth, 'Color', ModelColor);
    plot(tspan, PsiList_Analytical(:,idx(1)), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(1)), DataDrivenMarker, 'Color', DataDrivenColor);
    title(META.labels(idx(1)))
    ylabel("Position [m]")
    hold off
subplot(2,2,2);
    hold on
    plot(tspan, PsiList_Model(:,idx(2)), 'LineWidth', ModelLineWidth, 'Color', ModelColor);
    plot(tspan, PsiList_Analytical(:,idx(2)), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(2)), DataDrivenMarker, 'Color', DataDrivenColor);
    title(META.labels(idx(2)));
    hold off

subplot(2,2,3);
    hold on
    plot([0,tspan(end)], [0,0], 'Color', ReferenceColor);
    plot(tspan, PsiList_Analytical(:,idx(1))-PsiList_Model(:,idx(1)), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(1))-PsiList_Model(:,idx(1)), DataDrivenMarker, 'Color', DataDrivenColor);
    ylabel("Error [m]");
    ylim([-0.1, 0.15]);
    xlabel("Time [s]");
    hold off
subplot(2,2,4);
    hold on
    plot([0,tspan(end)], [0,0], 'Color', ReferenceColor);
    plot(tspan, PsiList_Analytical(:,idx(2))-PsiList_Model(:,idx(2)), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(2))-PsiList_Model(:,idx(2)), DataDrivenMarker, 'Color', DataDrivenColor);
    ylim([-0.1, 0.15]);
    xlabel("Time [s]");
    legend("Model", "Analytical", "Data-Driven");
    hold off

if save_figures
    figure_name = "x_propagation.png";
    exportgraphics(x_fig, figure_path + figure_name, 'Resolution', 600);
end


%% results for x'x propagation
idx = META.xx;
xTx_fig = figure('Position', [1921,-282,1070,599]);
% sgtitle("Propagation of x^Tx")
subplot(2,4,1);
    hold on
    plot(tspan, PsiList_Model(:,idx(1)), 'LineWidth', ModelLineWidth, 'Color', ModelColor);
    plot(tspan, PsiList_Analytical(:,idx(1)), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(1)), DataDrivenMarker, 'Color', DataDrivenColor);
    title(META.labels(idx(1)))
    ylabel("Magnitude [m^2]");
    hold off
subplot(2,4,2);
    hold on
    plot(tspan, PsiList_Model(:,idx(2)), 'LineWidth', ModelLineWidth, 'Color', ModelColor);
    plot(tspan, PsiList_Analytical(:,idx(2)), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(2)), DataDrivenMarker, 'Color', DataDrivenColor);
    title(META.labels(idx(2)));
    hold off
subplot(2,4,3);
    hold on
    plot(tspan, PsiList_Model(:,idx(3)), 'LineWidth', ModelLineWidth, 'Color', ModelColor);
    plot(tspan, PsiList_Analytical(:,idx(3)), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(3)), DataDrivenMarker, 'Color', DataDrivenColor);
    title(META.labels(idx(3)));
    hold off
subplot(2,4,4);
    hold on
    plot(tspan, PsiList_Model(:,idx(4)), 'LineWidth', ModelLineWidth, 'Color', ModelColor);
    plot(tspan, PsiList_Analytical(:,idx(4)), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(4)), DataDrivenMarker, 'Color', DataDrivenColor);
    title(META.labels(idx(4)));
    legend("Model", "Analytical", "Data-Driven");
    hold off

subplot(2,4,5);
    hold on
    plot([0,tspan(end)], [0,0], 'Color', ReferenceColor);
    plot(tspan, PsiList_Analytical(:,idx(1))-PsiList_Model(:,idx(1)), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(1))-PsiList_Model(:,idx(1)), DataDrivenMarker, 'Color', DataDrivenColor);
    ylabel("Error [m^2]");
    xlabel("Time [s]");
    hold off
subplot(2,4,6);
    hold on
    plot([0,tspan(end)], [0,0], 'Color', ReferenceColor);
    plot(tspan, PsiList_Analytical(:,idx(2))-PsiList_Model(:,idx(2)), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(2))-PsiList_Model(:,idx(2)), DataDrivenMarker, 'Color', DataDrivenColor);
    xlabel("Time [s]");
    hold off
subplot(2,4,7);
    hold on
    plot([0,tspan(end)], [0,0], 'Color', ReferenceColor);
    plot(tspan, PsiList_Analytical(:,idx(3))-PsiList_Model(:,idx(3)), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(3))-PsiList_Model(:,idx(3)), DataDrivenMarker, 'Color', DataDrivenColor);
    xlabel("Time [s]");
    hold off
subplot(2,4,8);
    hold on
    plot([0,tspan(end)], [0,0], 'Color', ReferenceColor);
    plot(tspan, PsiList_Analytical(:,idx(4))-PsiList_Model(:,idx(4)), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, PsiList_DataDriven(:,idx(4))-PsiList_Model(:,idx(4)), DataDrivenMarker, 'Color', DataDrivenColor);
    xlabel("Time [s]");
    hold off

if save_figures
    figure_name = "xTx_propagation.png";
    exportgraphics(xTx_fig, figure_path + figure_name, 'Resolution', 600);
end


%% results for d propagation
idx = META.d;
d_fig = figure('Position', [1921,-836,1070,599]);
% sgtitle("Propagation of O(x)")
subplot(2,4,1);
    hold on
    plot(tspan, sqrt(PsiList_Model(:,idx(1))), 'LineWidth', ModelLineWidth, 'Color', ModelColor);
    plot(tspan, sqrt(PsiList_Analytical(:,idx(1))), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, sqrt(PsiList_DataDriven(:,idx(1))), DataDrivenMarker, 'Color', DataDrivenColor);
    title("d(x,"+META.labels(idx(1))+")^{1/2}")
    ylabel("Distance [m]");
    hold off
subplot(2,4,2);
    hold on
    plot(tspan, sqrt(PsiList_Model(:,idx(2))), 'LineWidth', ModelLineWidth, 'Color', ModelColor);
    plot(tspan, sqrt(PsiList_Analytical(:,idx(2))), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, sqrt(PsiList_DataDriven(:,idx(2))), DataDrivenMarker, 'Color', DataDrivenColor);
    title("d(x,"+META.labels(idx(2))+")^{1/2}")
    hold off
subplot(2,4,3);
    hold on
    plot(tspan, sqrt(PsiList_Model(:,idx(3))), 'LineWidth', ModelLineWidth, 'Color', ModelColor);
    plot(tspan, sqrt(PsiList_Analytical(:,idx(3))), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, sqrt(PsiList_DataDriven(:,idx(3))), DataDrivenMarker, 'Color', DataDrivenColor);
    title("d(x,"+META.labels(idx(3))+")^{1/2}")
    hold off
subplot(2,4,4);
    hold on
    plot(tspan, sqrt(PsiList_Model(:,idx(4))), 'LineWidth', ModelLineWidth, 'Color', ModelColor);
    plot(tspan, sqrt(PsiList_Analytical(:,idx(4))), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, sqrt(PsiList_DataDriven(:,idx(4))), DataDrivenMarker, 'Color', DataDrivenColor);
    title("d(x,"+META.labels(idx(4))+")^{1/2}")
    legend("Model", "Analytical", "Data-Driven");
    hold off

subplot(2,4,5);
    hold on
    plot([0,tspan(end)], [0,0], 'Color', ReferenceColor);
    plot(tspan, sqrt(PsiList_Analytical(:,idx(1)))-sqrt(PsiList_Model(:,idx(1))), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, sqrt(PsiList_DataDriven(:,idx(1)))-sqrt(PsiList_Model(:,idx(1))), DataDrivenMarker, 'Color', DataDrivenColor);
    ylabel("Error [m]");
    xlabel("Time [s]");
    hold off
subplot(2,4,6);
    hold on
    plot([0,tspan(end)], [0,0], 'Color', ReferenceColor);
    plot(tspan, sqrt(PsiList_Analytical(:,idx(2)))-sqrt(PsiList_Model(:,idx(2))), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, sqrt(PsiList_DataDriven(:,idx(2)))-sqrt(PsiList_Model(:,idx(2))), DataDrivenMarker, 'Color', DataDrivenColor);
    xlabel("Time [s]");
    hold off
subplot(2,4,7);
    hold on
    plot([0,tspan(end)], [0,0], 'Color', ReferenceColor);
    plot(tspan, sqrt(PsiList_Analytical(:,idx(3)))-sqrt(PsiList_Model(:,idx(3))), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, sqrt(PsiList_DataDriven(:,idx(3)))-sqrt(PsiList_Model(:,idx(3))), DataDrivenMarker, 'Color', DataDrivenColor);
    xlabel("Time [s]");
    hold off
subplot(2,4,8);
    hold on
    plot([0,tspan(end)], [0,0], 'Color', ReferenceColor);
    plot(tspan, sqrt(PsiList_Analytical(:,idx(4)))-sqrt(PsiList_Model(:,idx(4))), AnalyticalMarker, 'Color', AnalyticalColor);
    plot(tspan, sqrt(PsiList_DataDriven(:,idx(4)))-sqrt(PsiList_Model(:,idx(4))), DataDrivenMarker, 'Color', DataDrivenColor);
    xlabel("Time [s]");
    hold off

if save_figures
    figure_name = "d_propagation.png";
    exportgraphics(d_fig, figure_path + figure_name, 'Resolution', 600);
end


%% robot variable for animation and distance
bernard = struct;
bernard.x = x0;
bernard.r = 0.25;
bernard.color = [0 0.4470 0.7410];

% animate(world, bernard, [], tspan(2), PsiList_Model(1,META.x), [], PsiList_Model(1,META.d(2:end)));
% animate(world, bernard, [], tspan(2), PsiList_Analytical(2,META.x), [], PsiList_Analytical(2,META.d(2:end)));


%% local functions
function [Psi_n] = KoopPropagate(Psi, u, K, world, META)

    [dPsix, dPsiu] = observables_partial(Psi(META.x), u, world);
    Psi_n = Psi(META.x)*dPsix*K + u*dPsiu*K;

end