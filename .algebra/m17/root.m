%% clean workspace
clean;


%% varaible setup
A = 10;
a = A/2;

Nt = 10;  % number of data points
m = 2;  % number of inputs (currently unused)
p = 4;  % number of observable functions on Psix
q = 6;  % number of observable functions in Psiu
b = 8;  % number of observable functions in h

PsiX = A*rand(p,Nt)-a;
PsiU = A*rand(q,Nt)-a;
h = A*rand(b,Nt)-a;

PsiData = [];
for i = 1:Nt
    PsiData = [PsiData, kron(PsiU(:,i), h(:,i))];
end
PsiData = [PsiX; PsiData];
PsiPlus = A*rand(p+b*q,Nt);

Ib = eye(b);
Kx = 1/2*(A*rand(p+b*q)-a);
Ku = 1/2*(A*rand(b)-a);


%% true error/cost calc.
Mu = Kblock(Ku, p, q, b);
true = norm(PsiPlus - Kx*Mu*PsiData);


%% vector form error/cost
Kxl = Kx(:,1:p);
Kxr = Kx(:,p:end);
d = vec(PsiPlus - Kxl*PsiX);

C = [];
for i = 1:Nt
    M
end



%% compare
test1 = 0;

fprintf("Error 1: %0.3s\n", norm(true - test1));
% fprintf("Error 2: %0.3s\n", norm(true - test2));
