import numpy as np

def vec(A):
    (n,m) = A.T.shape;
    return A.T.reshape(n*m,1);

# dimension variables and environment setup
m = 3;
p = 2;
q = 6;
b = 5;
Nt = 2;

print(p+q*b);

PsiX = np.random.rand(p,Nt);
PsiU = np.random.rand(q,Nt);
PsiH = np.random.rand(b,Nt);

Psi1 = np.empty( (q*b,Nt) );
for i in range(Nt):
    Psi1[:,i] = np.kron(PsiU[:,i], PsiH[:,i]);
Psi1 = np.vstack( (PsiX, Psi1) );
Psi2 = np.random.rand( p+q*b,Nt );

Kx = np.eye(p+q*b,p+q*b);
Ku = np.eye(b,b);

def Kblock(K):
    Ip = np.eye(p);
    Zpqb = np.zeros( (p,q*b) );
    M = np.vstack( (
        np.hstack( (Ip, Zpqb) ),
        np.hstack( (Zpqb.T, np.kron(np.eye(q), Ku)) )
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

    Psi2Left = vec(Kxl@PsiX);

    # print(vec(Psi2) - Psi2Left);

    c = 0;
    skip = p+q*b;
    Clist = np.empty( (Nt*(skip), b*b) );
    for k in range(Nt):
        s = 0;
        Mlist = np.empty( (skip,b*b) );
        for i in range(b):
            ei = np.array( [[1*(i==l)] for l in range(b)] );
            for j in range(b):
                ej = np.array( [[1*(j==l)] for l in range(b)] );
                M = np.kron(PsiU[:,k,None], ei@ej.T@PsiH[:,k,None]);
                # M = np.vstack( (PsiX[:,k,None], M) );
                M = np.vstack( (np.zeros( (p,1) ), M) );
                Mlist[:,s] = M[:,0];
                s += 1;
        Clist[c:c+skip,:] = Mlist;
        c += skip;

    print(Kx.shape, Kxl.shape, Kxr.shape, Kblock(Ku).shape);

    Psi2Right = Clist@vec( Ku );
    print(vec(Psi2).shape, Psi2Left.shape, Psi2Right.shape);

    return np.linalg.norm( vec(Psi2) - Psi2Left - Psi2Right );

# comparison
print( trueCost() );
print( vectCost() );