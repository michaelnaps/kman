import numpy as np

def vec(A):
    (n,m) = A.T.shape;
    return A.T.reshape(n*m,1);

# dimension variables and environment setup
m = 2;
p = 5;
q = 3;
b = 6;
Nt = 2;

PsiX = np.random.rand(p,Nt);
PsiU = np.random.rand(q,Nt);
PsiH = np.random.rand(b,Nt);

Psi1 = np.empty( (q*b,Nt) );
for i in range(Nt):
    Psi1[:,i] = np.kron(PsiU[:,i], PsiH[:,i]);
Psi1 = np.vstack( (PsiX, Psi1) );
Psi2 = np.random.rand(p+q*b,Nt);

Kx = np.random.rand(p+q*b, p+q*b);
Ku = np.random.rand(b,b);

def Kblock(K):
    M = np.vstack( (
        np.hstack( (np.eye(p), np.zeros( (p,q*b) )) ),
        np.hstack( (np.zeros( (q*b,p) ), np.kron(np.eye(q), Ku)) )
    ) );
    return M;

# true cost form
def trueCost():
    Mu = Kblock(Ku);
    PsiPlus = Kx@Mu@Psi1;
    return np.linalg.norm( Psi2 - PsiPlus );

# vectorized cost form
def vectCost():
    Kxl = Kx[:,:p];
    Kxr = Kx[:,p:];
    Psi2Right = vec(Kxl@PsiX);

    c = 0;
    Clist = np.empty( (Nt*(p+q*b), b*b) );
    for k in range(Nt):
        s = 0;
        Mlist = np.empty( (p+q*b,b*b) );
        for i in range(b):
            ei = np.array( [[1*(i==l)] for l in range(b)] );
            for j in range(b):
                ej = np.array( [[1*(j==l)] for l in range(b)] );
                M = np.kron(PsiU[:,k,None], ei@ej.T@PsiH[:,k,None])
                M = np.vstack( (PsiX[:,k,None], M) );
                Mlist[:,s] = M[:,0];
                s += 1;
        Clist[c:c+(p+q*b),:] = Mlist;
        c += p+q*b;

    Psi2Left = Clist@vec(Ku);
    return np.linalg.norm( vec(Psi2) - Psi2Left - Psi2Right );

# comparison
print( trueCost() );
print( vectCost() );