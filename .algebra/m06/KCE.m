clean;

%% obs function sizes
p = 2;
q = 2;
b = 2;


%% operators
Kx = sym([]);
for i = 1:p+q*b
    temp = sym([]);
    for j = 1:p+q*b
        temp = [temp, "p"+i+j];
    end
    Kx = [Kx; temp];
end

disp(Kx);

Ku = [];
for i = 1:b
    temp = sym([]);
    for j = 1:b
        temp = [temp, "b"+i+j];
    end
    Ku = [Ku; temp];
end

disp(Ku);

Ktemp = sym([
    eye(p), zeros(p,q*b);
    zeros(q*b,p), kron(eye(q), Ku);
]);

disp(Ktemp);

K = Kx*Ktemp;

disp(K);


%% observables
PsiX = sym([]);
for i = 1:p
    PsiX = [PsiX; "x"+i];
end

PsiU = sym([]);
for i = 1:q
    PsiU = [PsiU; "u"+i];
end

PsiH = sym([]);
for i = 1:b
    PsiH = [PsiH; "h"+i];
end

Psi = [PsiX; kron(PsiU,PsiH)];

disp(Psi);

disp(K*Psi);











