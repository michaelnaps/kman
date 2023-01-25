import numpy as np

class MetaVariable:
    def __init__(self, Nk):
        self.Nk = Nk;
        return;

def obsX(x):
    Nx = 4;
    Nk = 5;

    PsiX = np.zeros((Nk,1));
    PsiX[0:Nx,:] = x;
    PsiX[Nx,:] = [1];

    metaX = MetaVariable(Nk);
    metaX.x = [1,2,3,4];
    metaX.c = [5];

    return (PsiX, metaX);

def obsU(x):
    Nk = 5;
    (PsiU, _) = obsX(x);

    metaU = MetaVariable(Nk);
    metaU.xu = [1,2,3,4];
    metaU.cu = [5];

    return (PsiU, metaU);

def obsXU(x, u):
    Nx = 4
    Nu = 2;

    (PsiX, metaX) = obsX(x);
    (PsiU, metaU) = obsU(x);
    Nk = metaX.Nk + Nu*metaU.Nk;

    PsiXU = np.zeros((Nk,1));
    PsiXU[0:metaX.Nk,:] = PsiX;
    PsiXU[metaX.Nk:Nk,:] = np.kron(PsiU, u);

    metaXU = MetaVariable(Nk);
    return (PsiXU, metaXU);

def obsH(x, u):
    Nx = 4;
    Nu = 2;
    Nk = Nx + Nu + 1;
    metaH = MetaVariable(Nk);

    PsiH = np.zeros((Nk,1));
    PsiH = np.vstack((x,u, [1]));

    return (PsiH, metaH);

if __name__ == "__main__":
    x = np.array([[1],[2],[0],[-1]]);
    u = np.array([[5],[2]]);

    print(obsH(x,u));
