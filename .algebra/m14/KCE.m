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

true = kron(Psiu, Ku*h);

test1 = kron(Psiu, vec(Ku*h));

Ib = eye(b);
test2 = kron(Psiu, kron(h', Ib))*vec(Ku);

fprintf("Error 1: %0.3s\n", norm(true - test1));
fprintf("Error 2: %0.3s\n", norm(true - test2));

% KCEtrue = Kx*[Psix; kron(Psiu, Ku*h)];
% 
% KCEtest1 = Kx*[Psix; kron(Psiu, kron(h', Ib))];