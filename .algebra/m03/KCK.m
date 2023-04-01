clean;

m = 5;
p = 2;
q = 7;
b = 3;

psix = rand(p,1);
psiu = rand(q,m);
h = rand(b,1);

Kx = diag(rand(p+q,1));
Ku = rand(m,b);

true = Kx*[psix; psiu*Ku*h];

Ik = eye(q);
hTg = kron(h',psiu);

A = [eye(p), zeros(p,m*b*q); zeros(q,p), kron(Ku(:)',Ik)];

test1 = Kx*[psix; kron(Ku(:)',Ik)*hTg(:)];

K = Kx*A;
test2 = K*[psix; hTg(:)];

fprintf("Error 1: %0.3s\n", norm(true - test1));
fprintf("Error 2: %0.3s\n", norm(true - test2));