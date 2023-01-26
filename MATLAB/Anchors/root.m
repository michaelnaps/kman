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
B = [
    0, 0;
    0, 0;
    dt, 0;
    0, dt
];
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
tlist = 0:dt:T;
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
[~, metaX] = obsX(x0);
[~, metaU] = obsU(x0);
[~, metaXU] = obsXU(x0);
[~, metaH] = obsH([x0;u0]);

[~, meta] = obs(x0);


%% generate Ku
Xu = [xlist; zeros(2,Nt)];
Yu = [xlist; ulist];

Ku = Koopman(@(x)obsH(x), Xu, Yu, [x0;u0]);


%% generate Kx
Xxu = xlist(:,1:end-1);
Yxu = xlist(:,2:end);
[Kx, acc, ind, err] = Koopman(@(x)obsXU(x), Xxu, Yxu, x0);


%% create combined operator, K
m = Nu;
p = metaX.Nk;
q = metaU.Nk;
b = metaH.Nk;

K = Kx*[
    eye(p), zeros(p, q*b);
    zeros(q*b, p), kron(eye(q), Ku)
];


%% comparison data
% x0 = 2*rand(Nx,1) - 1;
psitest = NaN(meta.Nk,Nt);

psitest(:,1) = obs(x0);

for i = 1:Nt-1
    psitest(:,i+1) = K*psitest(:,i);
end


%% plot test results
figure(1)
subplot(1,2,1)
    hold on
    plot(tlist, xlist(1,:), 'b')
    plot(tlist, psitest(1,:), '--r')
    hold off
subplot(1,2,2)
    hold on
    plot(tlist, xlist(2,:), 'b')
    plot(tlist, psitest(2,:), '--r')
    hold off
% subplot(2,1,2)
%     hold on
%     plot(tlist, ulist, 'b')
%     plot(tlist, psitest(1:2,:), '--r')
%     hold off
