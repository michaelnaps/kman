%% clean workspace
clean;


%% varaible setup
A = 10;
a = A/2;

Nt = 10;  % number of data points
m = 2;  % number of inputs (currently unused)
p = 10;  % number of observable functions on Psix
q = 6;  % number of observable functions in Psiu
b = 8;  % number of observable functions in h

PsiX = A*rand(p,Nt)-a;
PsiU = A*rand(q,Nt)-a;
h = A*rand(b,Nt)-a;

Psi1 = [];
for i = 1:Nt
    Psi1 = [Psi1, kron(PsiU(:,i), h(:,i))];
end
Psi1 = [PsiX; Psi1];
Psi2 = A*rand(p+b*q,Nt);

Ib = eye(b);
Kx = 1/2*(A*rand(p+b*q)-a);
Ku = 1/2*(A*rand(b)-a);


%% true error/cost calc.
Mu = Kblock(Ku, p, q, b);
true = norm( vec( Psi2 - Kx*Mu*Psi1 ) );


%% vector form error/cost
Kxl = Kx(:,1:p);
Kxr = Kx(:,p:end);
PsiRight = vec(Kxl*PsiX);

C = [];
for k = 1:Nt
    Mlist = [];
    for i = 1:b
        ei = zeros(b,1);
        ei(i) = 1;
        for j = 1:b
            ej = zeros(b,1);
            e(j) = 1;
            Mc = kron(PsiU(:,i), ei*ej'*h(:,i));
            Mc = [PsiX(:,i); Mc];
            Mlist = [Mlist, Mc];
        end
    end
    C = [C; Mlist];
end
PsiLeft = C*vec(Ku);

test1 = norm(vec(Psi2) - PsiRight - PsiLeft);


%% compare
fprintf("Error 1: %0.3s\n", norm(true - test1));
% fprintf("Error 2: %0.3s\n", norm(true - test2));
