import sys
sys.path.insert(0, '/home/michaelnaps/prog/kman')
sys.path.insert(0, '/home/michaelnaps/prog/ode')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.path as path

import ode
import KMAN.KoopmanFunctions as kman
import KMAN.DataFunctions as data


# hyper parameter(s)
Nx = 2
Nu = 1
dt = 0.01


# set global output setting
np.set_printoptions(precision=3, suppress=True)

def modelFunc(X):
    mu = 1  a = 1
    x1 = X[0]  x2 = X[1]
    u = X[2]

    xn = X[:2].reshape(Nx,1) + dt*np.array( [
        [x2],
        [mu*(1 - x1**2)*x2 - x1 + a*np.sin(u)]
    ] )

    return np.vstack( (xn, u) )


def obsX(x=None):
    if x is None:
        meta = {'Nk':2}
        return meta
    Psi = x[:2]
    return Psi

def obsU(x=None):
    if x is None:
        meta = {'Nk':2}
        return meta
    Psi = np.vstack( ([1], x[1]) )
    return Psi

def obsH(x=None):
    if x is None:
        meta = {'Nk':3}
        return meta
    Psi = np.vstack( (x[2], np.sin(x[2]), x[0]**2) )
    return Psi


if __name__ == "__main__":
    # initial state parameters
    N0 = 10
    X0 = 10*np.random.rand(Nx+Nu,N0) - 5


    # create model data
    T = 10
    Nt = round(T/dt) + 1

    tList = np.array( [[i*dt for i in range(Nt)]] )

    xTrain, _ = data.generate_data(tList, modelFunc, X0)


    # observables for TRAINING
    def obsTrain(x=None):
        if x is None:
            meta = {'Nk': obsX()['Nk'] + obsU()['Nk']*obsH()['Nk']}
            return meta

        PsiX = obsX(x)
        PsiU = obsU(x)
        PsiH = obsH(x)

        Psi = np.vstack( (PsiX, np.kron(PsiU, PsiH)) )

        return Psi


    # organize data
    xData = xTrain[:,:Nt-1]
    yData = xTrain[:,1:Nt]

    X = data.stack_data(xData, N0, Nx+Nu, Nt-1)
    Y = data.stack_data(yData, N0, Nx+Nu, Nt-1)

    kvar = kman.KoopmanOperator(obsTrain)
    K = kvar.edmd(X, Y, X0)
    Kup = K[:Nx+Nu,:]

    print('K:', kvar.err)
    print(K, '\n')

    print('Kup:')
    print(Kup, '\n')


    # define observable function FOR IMPLEMENTATION
    def obsImplm(x=None):
        Nk = obsTrain()['Nk']
        if x is None:
            meta = {'Nk':Nk}
            return meta

        PsiX = x[:Nx].reshape(Nx,1)
        PsiU = np.vstack( ([1], x[1]) )
        PsiH = obsH(x)

        Psi = np.vstack( (PsiX, np.kron(PsiU, PsiH)) )

        return Psi.reshape(Nk,1)


    # compare koopman to real model
    Nk = obsTrain()['Nk']
    Psi0 = X0[:,0].reshape(Nx+Nu,1)
    xComp = xTrain[:Nx,:]

    koopFunc = lambda x: Kup@obsImplm(x)
    PsiTest, _ = data.generate_data(tList, koopFunc, Psi0)
    xTest = PsiTest[:Nx,:]


    # plot comparison
    fig, axs = plt.subplots(2,1)

    axs[0].plot(xTrain[0], xTrain[1])
    axs[0].plot(xTest[0], xTest[1], linestyle='--')
    axs[0].axis('equal')

    axs[1].plot(tList[0], PsiTest[2,:])
    axs[1].axis('equal')

    plt.show()
