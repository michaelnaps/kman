import numpy as np

np.set_printoptions(precision=2, suppress=True);

Nt=10
NkH=2 #nb of measurements from the robot
NkU=4
NkX=4
Nu=3 #nb of inputs for the robot

h=np.random.rand(NkH, Nt)
PsiX=np.random.rand(NkX, Nt)
Psin=np.random.rand(NkX+NkU*NkH, Nt)
PsiU=np.random.rand(NkU, Nt)

Ku=np.random.rand(NkH, NkH)
Kx=np.random.rand(NkX+NkH*NkU, NkX+NkH*NkU)

def vec(A):
    (n,m) = A.shape
    return A.reshape(n*m,1)

def Kblock(K):
    Kb = np.vstack( (
        np.hstack( (np.eye(NkX), np.zeros( (NkX, NkH*NkU) )) ),
        np.hstack( (np.zeros( (NkH*NkU, NkX) ), np.kron(np.eye(NkU), K)) )
    ) );
    return Kb;

def cost_matrix_form(Ku):
    Mu=Kblock(Ku);
    PsiUH = np.empty( (NkU*NkH, Nt) ) # matrix where each column is PsiU kron h
    for i_data in range(Nt):
        PsiUH[:,i_data]=np.kron(PsiU[:,i_data], h[:,i_data])
    return np.linalg.norm( Psin - Mu@np.vstack( (PsiX,PsiUH) ) )

def cost_vector_form(Ku):
    Kxleft=Kx[:,:NkX]
    Kxright=Kx[:,NkX:]
    d=vec(Psin - Kxleft@PsiX)
    C_list=[]
    for i_data in range(Nt):
        M_i_data_list=[]
        for p in range(NkH):
            for q in range(NkH):
                epq=np.zeros( (NkH,NkH) )
                epq[p,q] = 1
                M_i_data_column=np.kron(PsiU[:,i_data],epq@h)
                M_i_data_list.append(M_i_data_column)
        M_i_data=np.hstack(M_i_data_list)
        C_list.append(M_i_data)
    C=np.vstack(C_list)
    print(Ku)clear

    print(C.shape, vec(Ku).shape, d.shape)
    return np.linalg.norm(C@vec(Ku)-d)

# print( cost_matrix_form(Ku) );
print( cost_vector_form(Ku) );