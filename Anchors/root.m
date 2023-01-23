%% clean workspace and set path
clean;

addpath ../KoopFunctions
addpath ../DataFunctions
addpath ../PlotFunctions


%% create model/control equations
dt = 0.1;
xg = [1;-1;0;0];

c = 0.25;

A = [
    1, 0, dt, 0;
    0, 1, 0, dt;
    0, 0, 1-c, 0;
    0, 0, 0, 1-c;
];
B = [0, 0; 0, 0; dt, 0; 0, dt];
C = [
    10, 0, 2.5, 0;
    0, 10, 0, 2.5;
];

model = @(x,u) A*x + B*u;
control = @(x) C*(xg - x);


%% grab anchors
load anchors anchors;


%% initial conditions
Nx = 4;
Nu = 2;

x0 = [0;0;0;0];
u0 = [0;0];


%% test system model
T = 10;
tlist = 0:dt:10;
Nt = length(tlist);

xlist = NaN(Nx, Nt);
ulist = NaN(Nu, Nt);

xlist(:,1) = x0;
ulist(:,1) = u0;

for i = 1:Nt-1
    ulist(:,i+1) = control(xlist(:,i));
    xlist(:,i+1) = model(xlist(:,i), ulist(:,i+1));
end


%% get observable function meta-data
[~, metaH] = obsH([x0;u0]);
[~, metaX] = obsX(x0);
[~, metaU] = obsU(x0);
[~, metaXU] = obsXU(x0, u0);


%% generate Ku
Xu = [xlist; zeros(2,Nt)];
Yu = [xlist; ulist];

Ku = Koopman(@(x)obsH(x), Xu, Yu, [x0;u0]);




















