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
T = 5;
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


%% generate Kx
Xxu = xlist(:,1:end-1);
Yxu = xlist(:,2:end);
[Kx, acc, ind, err] = KoopmanWithControl(@(x,u)obsXU(x,u), Xxu, Yxu, x0, ulist);


%% create combined operator, K
m = Nu;
p = metaX.Nk;
q = metaU.Nk;
b = metaH.Nk;

K = Kx*[
    eye(p), zeros(p, m*q*b);
    zeros(q, p), kron(vec(Ku(metaH.u,:))', eye(q))
];


%% comparison data
x0 = 2*rand(Nx,1) - 1;

xtest = NaN(Nx,Nt);
utest = NaN(Nu,Nt);

xtest(:,1) = x0;
utest(:,end) = u0;

for i = 1:Nt-1
    utest(:,i) = Ku(metaH.u,:)*obsH([xtest(:,i);u0]);

    Psi = K*obs(xtest(:,i));
    xtest(:,i+1) = Psi(metaXU.x);
end


%% plot test results
figure(1)
subplot(2,2,1)
    hold on
    plot(tlist, xlist(1,:), 'b')
    plot(tlist, xtest(1,:), '--r')
    hold off
subplot(2,2,2)
    hold on
    plot(tlist, xlist(2,:), 'b')
    plot(tlist, xtest(2,:), '--r')
    hold off
subplot(2,1,2)
    hold on
    plot(tlist, ulist, 'b')
    plot(tlist, utest, '--r')
    hold off
















