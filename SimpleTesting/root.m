clean;

%% add paths
addpath /home/michaelnaps/prog/kman/MATLAB/KoopFunctions
addpath /home/michaelnaps/prog/kman/MATLAB/DataFunctions

%% construct model
Nx = 4;
Nu = 2;

dt = 0.1;
A = [
    1, 0, dt, 0;
    0, 1, 0, dt;
    0, 0, 1, 0;
    0, 0, 0, 1;
];
B = [
    0, 0;
    0, 0;
    dt, 0;
    0, dt;
];

model = @(x,u) A*x + B*u;

%% simulation variables
T = 10;  Nt = round(T/dt)+1;
tTrain = 0:dt:T;

%% generate list of randomly assorted u
x0 = 10*rand(4,1) - 5;

uRand = 2*rand(Nu, Nt-1) - 1;
xTrain = NaN(Nx, Nt);

xTrain(:,1) = x0;

for i = 1:Nt-1
    xTrain(:,i+1) = model(xTrain(:,i), uRand(:,i));
end

%% plot the trajectories
figure(1)
    subplot(2,1,1)
    hold on
        plot(xTrain(1,:), xTrain(2,:))
    hold off
    subplot(2,1,2)
    hold on
        plot(uRand')
    hold off

%% construct Kx data matrices
X = [xTrain(:,1:Nt-1); uRand];
Y = [xTrain(:,2:Nt); uRand];

Kx = Koopman(@(X)obs(X), X, Y, [x0;uRand(:,1)]);













