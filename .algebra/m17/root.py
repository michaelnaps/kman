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

PsiX = np.random.rand(p,Nt);
PsiU = np.random.rand(q,Nt);
PsiH = np.random.rand(b,Nt);

Psi1 = np.empty( (q*b,Nt) );
for i in range(Nt):
    Psi1[:,i] = np.kron(PsiU[:,i], PsiH[:,i]);
Psi1 = np.vstack( (PsiX, Psi1) );
Psi2 = np.random.rand( p+q*b,Nt );

Kx = np.eye(p+q*b,p+q*b);
Ku = np.random.rand(b,b);

print(PsiX.shape, PsiU.shape, PsiH.shape);
print(Psi1.shape, Psi2.shape);
print(Kx.shape, Ku.shape);

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
    return np.linalg.norm( Psi2 - Kx@Kblock(Ku)@Psi1 );

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
                M = np.kron(PsiU[:,k,None], ej@ei.T@PsiH[:,k,None]);
                # M = np.vstack( (PsiX[:,k,None], M) );
                M = np.vstack( (np.zeros( (p,1) ), M) );
                Mlist[:,s] = M[:,0];
                s += 1;
        Clist[c:c+skip,:] = Mlist;
        c += skip;


    print(Kxl.shape, Kxr.shape);
    print(vec(Ku).shape, vec(Kxl).shape, vec(Kxr).shape);
    print(Clist.shape);

    Psi2Cont = Clist@vec( Ku );
    return np.linalg.norm( vec(Psi2) - Psi2Left - Psi2Cont );

# comparison
print( trueCost() );
print( vectCost() );