%% clean/setup workspace
clean;

set(groot, 'DefaultLineLineWidth', 1.5);

addpath /home/michaelnaps/prog/kman/MATLAB/KoopFunctions
addpath /home/michaelnaps/prog/kman/MATLAB/DataFunctions


%% create model/control equations
dt = 0.1;
xg = [0; 0];

Nx = 2;
Nu = 1;

A = [1, dt; 0, dt];
B = [0; dt];
C = [10, 2.5];

model = @(x,u) A*x + B*u;
control = @(x) C*(xg - x);


%% initial state
x0 = 10*rand(Nx,1) - 5;


%% test run
T = 10;
Nt = round(T/dt)+1;

tlist = NaN(1,Nt);
ulist = NaN(Nu,Nt);
xlist = NaN(Nx,Nt);

tlist(1) = 0;
ulist(:,end) = 0;
xlist(:,1) = x0;

for i = 1:Nt-1
    tlist(i+1) = i*dt;
    ulist(i) = control(xlist(:,i));
    xlist(:,i+1) = model(xlist(:,i), ulist(i));
end


%% generate constant data
uconst = -3;

tconst = tlist;
uconst = kron(ones(1,Nt), uconst);
xconst = NaN(Nx,Nt);

xconst(:,1) = x0;

for i = 1:Nt-1
    xconst(:,i+1) = model(xlist(:,i), uconst(i));
end


%% meta data
[~, metaX] = obs_x(x0);
[~, metaU] = obs_u(x0);
[~, metaXU] = obs_xu([x0;0]);
[~, metaH] = obs_h([x0;0]);
[~, meta] = obs(x0);


%% generate Kx
Xxu = [xconst(:,1:end-1); uconst(1:end-1)];
Yxu = [xconst(:,2:end); uconst(2:end)];
Kx = Koopman(@(X)obs_xu(X), Xxu, Yxu, [x0;0]);

disp("Kx");
disp(Kx);

%% generate Ku
h = @(x) obs_h(x);

Xu = [xlist; zeros(1,length(xlist))];
Yu = [xlist; ulist];

Ku = Koopman(h, Xu, Yu, [x0;0]);

disp("Ku");
disp(Ku);

% set obs_h to only take x in R(2)
h = @(x) obs_h([x;0]);


%% create combined operator, K
m = 1;
p = metaX.Nk;
q = metaU.Nk;
b = metaH.Nk;

K = Kx*[
    eye(p), zeros(p, q*b);
    zeros(q*b, p), kron(eye(q), Ku)
];

disp("K");
disp(K);


%% comparison data
x0 = 4*rand(Nx,1) - 2;

utest = NaN(1,Nt);
xtest = NaN(2,Nt);
psitest = NaN(meta.Nk,Nt);

utest(end) = 0;
xtest(:,1) = x0;
psitest(:,1) = obs_xu([x0;0]);

for i = 1:Nt-1
    utest(i) = Ku(metaH.uh,:)*h(xtest(:,i));

    psitest(:,i+1) = Kx*psitest(:,i);
    xtest(:,i+1) = psitest(metaXU.x,i+1);
end


%% plot test results
figure(1)
subplot(2,2,1)
    hold on
    plot(tlist, xlist(1,:), 'b')
    plot(tlist, xtest(1,:), '--r')
    title("x1")
    hold off
subplot(2,2,2)
    hold on
    plot(tlist, xlist(2,:), 'b')
    plot(tlist, xtest(2,:), '--r')
    title("x2")
    hold off
subplot(2,1,2)
    hold on
    plot(tlist, ulist, 'b')
    plot(tlist, utest, '--r')
    title("u")
    hold off


