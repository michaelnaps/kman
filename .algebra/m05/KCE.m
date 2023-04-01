clean;

m = 2;  % number of inputs (currently unused)
p = 3;  % number of observable functions on Psix
q = 7;  % number of observable functions in Psiu
b = 3;  % number of observable functions in h

Psix = rand(p,1);
Psiu = rand(q,1);
h = rand(b,1);

Kx = eye(p+b*q);
Ku = rand(b);

true = Kx*[Psix; kron(Psiu, Ku*h)];

Iq = eye(q);

test1 = Kx*[Psix; kron(eye(q), Ku)*kron(Psiu, h)];

A = [eye(p), zeros(p,q*b); zeros(q*b,p), kron(Iq, Ku)];
K = Kx*A;

test2 = K*[Psix; kron(Psiu, h)];

fprintf("Error 1: %0.3s\n", norm(true - test1));
fprintf("Error 2: %0.3s\n", norm(true - test2));