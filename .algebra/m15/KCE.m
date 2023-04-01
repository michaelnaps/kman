clean;

A = 20;
a = A/2;

m = 2;  % number of inputs (currently unused)
p = 3;  % number of observable functions on Psix
q = 7;  % number of observable functions in Psiu
b = 3;  % number of observable functions in h

Psix = A*rand(p,1)-a;
Psiu = A*rand(q,1)-a;
h = A*rand(b,1)-a;

Kx = eye(p+b*q);
Ku = rand(b);

Kblock = [
    eye(p), zeros(p, q*b);
    zeros(q*b, p), kron(eye(q), Ku)
];
Psiblock = [Psix; kron(Psiu, h)];

true = Kx*Kblock*Psiblock;

test1 = kron(Kx, Psiblock')*vec(Kblock);

Ib = eye(b);
test2 = kron(Psiblock'*Kblock', eye(p+b*q))*vec(Kx);

fprintf("Error 1: %0.3s\n", norm(true - test1));
fprintf("Error 2: %0.3s\n", norm(true - test2));
