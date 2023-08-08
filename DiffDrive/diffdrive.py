# script for model equation testing
import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/mpc')
sys.path.insert(0, expanduser('~')+'/prog/kman')

import numpy as np

from KMAN.Operators import *
import MPC.Optimizer as mpc
import MPC.Vehicle2D as vhc

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.path as path


# print precision
np.set_printoptions(precision=5, suppress=True)


# functions for MPC
def model(x, u):
    xn = np.array( [
        x[0] + dt*math.cos(x[2])*(u[0] + u[1]),
        x[1] + dt*math.sin(x[2])*(u[0] + u[1]),
        x[2] + dt*1/R*(u[0] - u[1])
    ] )
    return xn

def cost(xlist, ulist):
    # gain parameters
    TOL = 1e-6
    kx = 1
    ko = 1

    # calculate cost of current input and state
    C = np.array( [0] )
    for i, x in enumerate(xlist):
        gx = (x[0] - xd[0])**2 + (x[1] - xd[1])**2
        go = (x[2] - xd[2])**2
        C = C + kx*gx + ko*go

    return C


# plot comparisons
def posTrackingNoControl(tList, kxvar, x0ref, uref):
    # dim variables
    NkX = obsX()['Nk']
    NkU = obsU()['Nk']

    # evaluate the behavior of Kx with remeasurement function
    dModel1 = lambda x: np.array( model(x,uref,None) ).reshape(Nx,1)
    kModel1 = lambda Psi: kxvar.K@rmes(Psi)
    def rmes(Psi):
        # tweaks for initial tests
        x = Psi[:Nx].reshape(Nx,1)
        u = uref
        X = np.vstack( (x,u) )

        PsiX = Psi[:NkX].reshape(NkX,1)
        PsiU = Psi[NkX:].reshape(NkU,1)
        PsiH = obsH(X)

        Psin = np.vstack( (PsiX, np.kron(PsiU, PsiH)) )
        return Psin

    # test data generation
    Psi0 = obsXU( np.vstack( (x0ref, uref) ) )
    PsiTest = generate_data(tList, kModel1, Psi0)[0]
    xTest = generate_data(tList, dModel1, x0ref)[0]

    return xTest, PsiTest

def plotcomp(tList, xTest, PsiTest, save=0):
    # plot test results
    fig, axs = plt.subplots(1,2)

    axs[0].plot(xTest[0], xTest[1], label='Model')
    axs[0].plot(PsiTest[0], PsiTest[1], linestyle='--', label='KCE')

    axs[0].set_xlabel('$x_1$')
    axs[0].set_ylabel('$x_2$')
    axs[0].axis('equal')
    axs[0].grid(1)
    axs[0].legend()

    # evaluate error
    Ne = tList.shape[1] - 1
    axs[1].plot([tList[0][0], tList[0][Ne]], [0,0], color='r', linestyle='--')
    axs[1].plot(tList[0][:Ne], PsiTest[0,:Ne]-xTest[0,:Ne], label='$x_1$')
    axs[1].plot(tList[0][:Ne], PsiTest[1,:Ne]-xTest[1,:Ne], label='$x_2$')
    axs[1].plot(tList[0][:Ne], PsiTest[2,:Ne]-xTest[2,:Ne], label='$x_3$')

    axs[1].set_ylabel('Error')
    axs[1].set_xlabel('Time')
    axs[1].set_ylim( (-1,1) )
    axs[1].grid(1)
    axs[1].legend()

    fig.tight_layout()

    # save results
    if save:
        figRes.savefig('/home/michaelnaps/prog/kman/.figures/donald.png', dpi=600)
        figError.savefig('/home/michaelnaps/prog/kman/.figures/donaldError.png', dpi=600)


# hyper parameter(s)
pi = math.pi
PH = 10
kl = 2
Nx = 3
Nu = 2
R = 1/2  # robot-body radius
dt = 0.001
alpha = 0.01

# initialize states
x0 = np.zeros( (Nx, 1) )
xd = np.zeros( (Nx, 1) )
uinit = np.zeros( (Nu*PH, 1) )

# create MPC class variable
dt_mpc = 0.01
max_iter = 100
simvar = vhc.Vehicle2D( x0[:2] )
mvar = mpc.ModelPredictiveControl( model, cost,
    P=PH, k=kl, Nx=Nx, Nu=Nu, dt=dt_mpc, cost_type='horizon' )
mvar.setStepSize( alpha )
mvar.setMaxIter( max_iter )


# observable functions
def obsX(x=None):
    if x is None:
        meta = {'Nk':Nx}
        return meta
    PsiX = x
    return PsiX

def obsU(x=None):
    if x is None:
        Ntrig = 2
        meta = {'Nk':Ntrig+1}
        return meta
    PsiU = np.vstack( (np.cos(x[2]), np.sin(x[2]), [1]) )
    return PsiU

def obsXU(X=None):
    if X is None:
        meta = {'Nk':obsX()['Nk']+obsU()['Nk']}
        return meta

    x = X[:Nx].reshape(Nx,1)
    u = X[Nx:].reshape(Nu*PH,1)

    PsiX = obsX(x)
    PsiU = obsU(x)

    PsiXU = np.vstack( (PsiX, PsiU) )
    return PsiXU

def obsH(X=None):
    if X is None:
        meta = {'Nk':2*Nu*PH+1}
        return meta

    u = X[:Nu*PH].reshape(Nu*PH,)
    x = X[Nu*PH:].reshape(Nx,)

    mvar.setObjectiveFunction( mvar.costFunctionGenerator( x0 ) )
    g = mvar.grad( u )

    PsiH = np.vstack( (u[:,None], g[:,None], [1]) )
    return PsiH

def obsXUH(X=None):
    if X is None:
        meta = {'Nk':obsX()['Nk']+obsU()['Nk']*obsH()['Nk']}
        return meta

    x = X[:Nx].reshape(Nx,1)
    u = X[Nx:].reshape(Nu*PH,1)

    PsiX = obsX(x)
    PsiU = obsU(x)
    PsiH = obsH(X)

    PsiXUH = np.vstack( (PsiX, np.kron(PsiU, PsiH)) )
    return PsiXUH

def obsUG(X=None):
    if X is None:
        meta = {'Nk': 2*Nu*PH}
        return meta

    x0  = X[Nu*PH:]
    uPH = X[:Nu*PH]
    mvar.setObjectiveFunction( mvar.costFunctionGenerator( x0 ) )

    PsiU = uPH
    PsiG = mvar.grad( uPH )
    PsiUG = np.vstack( (PsiU, PsiG) )

    return PsiUG

def obsUGX(X=None):
    if X is None:
        meta = {'Nk': (2*Nu*PH+1)*5*Nx}
        return meta

    x0  = X[Nu*PH:]
    x_01 = np.vstack( ([1], x0[:2]) )
    xTrg = np.vstack( ([1], [np.cos(x0[2])**i for i in range(1,3)], [np.sin(x0[2])**i for i in range(1,3)]) )

    PsiUG = np.vstack( (obsUG(X), [1]) )
    PsiUGX = np.kron( PsiUG, np.kron( x_01, xTrg ) )
    return PsiUGX