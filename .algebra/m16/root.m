clean;

A = 10;
a = A/2;

m = 2;  % number of inputs (currently unused)
p = 4;  % number of observable functions on Psix
q = 6;  % number of observable functions in Psiu
b = 8;  % number of observable functions in h

Psix = A*rand(p,1)-a;
Psiu = A*rand(q,1)-a;
h = A*rand(b,1)-a;

Ib = eye(b);
Kx = 1/2*(A*rand(p+b*q)-a);
Ku = 1/2*(A*rand(b)-a);

true = kron(Psiu, Ku*h);

[Mu1, vKu1] = KoopTest1(Ku, Psiu, h);
test1 = Mu1;

test2 = KoopTest2(Ku, Psiu, h);

fprintf("Error 1: %0.3s\n", norm(true - test1));
fprintf("Error 2: %0.3s\n", norm(true - test2));

%% kce new function
function [Mu, vKu] = KoopTest1(Ku, Psiu, h)

    Nk = length(Ku);
    Mu = zeros( size( kron(Psiu, Ku*h) ) );
    vKu = NaN(Nk*Nk,1);
    
    k = 1;
    for j = 1:Nk

        ej = zeros(Nk,1);
        ej(j) = 1;
        
        for i = 1:Nk

            ei = zeros(Nk,1);
            ei(i) = 1;

%             disp('______________________')
%             disp(kron(Psiu, ei*ej'*h)*Ku(i,j))

            vKu(k) = Ku(i,j);
            Mu = Mu + kron(Psiu, ei*ej'*h);
            k = k + 1;

        end

    end

end

function [Ku] = KoopTest2(Ku, Psiu, h)

    [Mu, vKu] = KoopTest1(Ku, Psiu, h);
    Ku = Mu.*vKu;

end