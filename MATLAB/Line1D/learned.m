%% clean/setup workspace
clean;

set(groot, 'DefaultLineLineWidth', 1.5);

addpath /home/michaelnaps/prog/kman/MATLAB/KoopFunctions
addpath /home/michaelnaps/prog/kman/MATLAB/DataFunctions


%% create model/control equations
dt = 0.1;
xg = [0; 0];

A = [1, dt; 0, dt];
B = [0; dt];
C = [10, 2.5];

model = @(x,u) A*x + B*u;
control = @(x) C*(xg - x);


%% initial state
x0 = 10*rand(2,1) - 5;


%% test run
T = 10;
N = round(T/dt);

tlist = NaN(1,N+1);
ulist = NaN(1,N+1);
xlist = NaN(2,N+1);

tlist(1) = 0;
ulist(:,end) = 0;
xlist(:,1) = x0;

for i = 1:N
    tlist(i+1) = i*dt;
    ulist(i) = control(xlist(:,i));
    xlist(:,i+1) = model(xlist(:,i), ulist(i));
end


%% meta data
[~, metaX] = obs_x(x0);
[~, metaU] = obs_u(x0);
[~, metaXU] = obs_xu(x0, 0);
[~, metaH] = obs_h([x0;0]);
[~, meta] = obs(x0);


%% generate Kx
Xxu = xlist(:,1:end-1);
Yxu = xlist(:,2:end);
Kx = KoopmanWithControl(@(x,u)obs_xu(x,u), Xxu, Yxu, x0, ulist);

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
    eye(p), zeros(p, m*q*b);
    zeros(q, p), kron(eye(q), Ku(metaH.u,:))
];

disp("K");
disp(K);


%% comparison data
% x0 = 4*rand(2,1)-2;

utest = NaN(1,N+1);
xtest = NaN(2,N+1);

utest(end) = 0;
xtest(:,1) = x0;

for i = 1:N
    utest(i) = Ku(metaH.u,:)*h(xtest(:,i));

    Psi_xuh = K*obs(xtest(:,i));
    xtest(:,i+1) = Psi_xuh(metaXU.x);
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


