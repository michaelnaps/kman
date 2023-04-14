import numpy as np
np.set_printoptions(precision=3, suppress=True);

def vec(A):
    (n,m) = A.T.shape;
    return A.T.reshape(n*m,1);

# dimension variables and environment setup
m = 2;
p = 2;
q = 3;
b = 2;
Nt = 2;

def Kblock(K):
    Ip = np.eye(p);
    Zpqb = np.zeros( (p,q*b) );
    M = np.vstack( (
        np.hstack( (Ip, Zpqb) ),
        np.hstack( (Zpqb.T, np.kron(np.eye(q), Ku)) )
    ) );
    return M;

PsiX = np.random.rand(p,Nt);
PsiU = np.random.rand(q,Nt);
PsiH = np.random.rand(b,Nt);

PsiUH = np.empty( (q*b,Nt) );
for i in range(Nt):
    PsiUH[:,i] = np.kron(PsiU[:,i], PsiH[:,i]);
Psi1 = np.vstack( (PsiX, PsiUH) );
Psi2 = np.random.rand( p+q*b,Nt );

Kx = np.random.rand(p+b*q,p+b*q)
Ku = np.random.rand(b,b);

# print(Kx); print(Ku);

print(PsiX.shape, PsiU.shape, PsiH.shape);
print(Psi1.shape, Psi2.shape);
print(Kx.shape, Ku.shape, Kblock(Ku).shape);

# true cost form
def trueCost():
    PsiDiff = vec( Psi2 - Kx@Kblock(Ku)@Psi1 );
    print(PsiDiff[:,0]);
    return np.linalg.norm( PsiDiff );

# vectorized cost form
def vectCost():
    Kxl = Kx[:,:p];
    Kxr = Kx[:,p:];

    Kxt = Kx[:p,:];
    Kxb = Kx[p:,:];

    # Psi2Top  = vec( Kxl@Kblock(Ku)[:,:p]@PsiX );
    Psi2Top = vec( Kxl@PsiX );
    # Psi2Right = vec( Kxr@PsiUH );

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
    print((Kxb@Kblock(Ku)[:,p:]).shape);
    print(vec(Ku).shape, vec(Kxl).shape, vec(Kxr).shape);
    print(Clist.shape);

    Psi2Bot = Clist@vec( Ku );
    # Psi2Bot = Clist@vec( Kxb@Kblock(Ku)[:,p:] );

    PsiDiff = vec(Psi2) - Psi2Top - Psi2Bot;
    print(PsiDiff[:,0]);

    return np.linalg.norm( PsiDiff );

def splitCost():
    Kxl = Kx[:,:p];
    Kxr = Kx[:,p:];

    PsiL = Kxl@PsiX;

    PsiR = np.empty( (p+q*b,Nt) );
    for i in range(Nt):
        # PsiR[:,i] = Kxr@( np.kron( PsiU[:,i,None], Ku@PsiH[:,i,None] ) )[:,0];
        PsiR[:,i] = (np.kron( np.kron( PsiU[:,i,None], PsiH[:,i,None] ).T, Kxr ) @ vec( np.kron(np.eye(q), Ku) ))[:,0];

    # PsiR = np.kron( np.kron( PsiU, PsiH ).T, Kxr ) @ vec( np.kron( np.eye(q), Ku ) );

    print(Psi2.shape, PsiL.shape, PsiR.shape)

    PsiDiff = Psi2 - PsiL - PsiR;

    print(vec( PsiDiff )[:,0]);

    return np.linalg.norm( PsiDiff );

# comparison
print('-----')
trueNorm = trueCost();

print('-----');
vectNorm = vectCost();

print('-----');
splitNorm = splitCost();

print('-----');
print( 'true: ', trueNorm );
print( 'vect: ', vectNorm );
# print( 'split:', splitNorm );