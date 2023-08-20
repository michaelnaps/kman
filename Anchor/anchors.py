import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman')
sys.path.insert(0, expanduser('~')+'/prog/geom')

import numpy as np
import matplotlib.pyplot as plt

from KMAN.Operators import *
from GEOM.Vehicle2D import *

import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.path as path

# set global output setting
np.set_printoptions(precision=3, suppress=True)


# hyper paramter(s)
epsilon = 0.25
delta = 1.00
dt = 0.01
Nx = 2
Nu = 2
Na = 4
# aList = np.array( [[10, 12, -15],[10, -7, -13]] )
aList = np.array( [[10, -10, 10], [10, 10, -10]] )
# aList = np.array( [[10, 15, -10, 10], [15, 10, 10, -10]] )

# Na = 5
# aList = np.array( [[10, 10, -10, -10, -5],[10, -10, -10, 10, -5]] )

# for plotting
aColor = 'indianred'
mColor = 'royalblue'
kColor = 'yellowgreen'
x1Color = 'indianred'
x2Color = 'orange'


# plot functions
def plotAnchors(fig, axs, radius=0.5):
    axs.scatter(aList[0][0], aList[0][1],
        color=aColor,
        marker='o', label='Anchor(s)')
    for a in aList.T:
        axs.plot(a[0], a[1])  # to shape axis around anchors
        circEntity = plt.Circle(a, radius,
            facecolor=aColor, edgecolor='k', zorder=100)
        axs.add_artist( circEntity )
    return fig, axs


# model and control functions
def model(x, u):
    return x + dt*u

def control(x):
    xg = [ [-3], [4] ]
    return xg - x

def noise(alpha, shape):
    return 2*alpha*np.random.rand(shape[0], shape[1]) - alpha

def anchorMeasure(x):
    da = np.empty( (Na,1) )
    for i, a in enumerate(aList.T):
        da[i] = np.linalg.norm(x - a[:,None])
    return da


# observable functions PsiX, PsiU, PsiH
def obsX(X=None):
    if X is None:
        meta = {'Nk':Nx}
        return meta
    PsiX = X[:Nx].reshape(Nx,1)
    return PsiX

def obsU(X=None):  # for training purposes only
    if X is None:
        meta = {'Nk':Nu}
        return meta

    PsiU = X[Nx:].reshape(Nu,1)
    return PsiU

def obsXU(X=None):
    if X is None:
        meta = {'Nk':obsX()['Nk']+obsU()['Nk']}
        return meta
    PsiX = obsX(X)
    PsiU = obsU(X)
    PsiXU = np.vstack( (PsiX, PsiU) )
    return PsiXU

def obsH(X=None):
    if X is None:
        meta = {'Nk':Na+1}
        return meta

    x = X[:Nx].reshape(Nx,1)
    PsiH = np.vstack( (anchorMeasure(x)**2, [1]) )

    return PsiH

def obsXUH(X=None):
    if X is None:
        meta = {'Nk':obsX()['Nk']+1*obsH()['Nk']}
        return meta

    PsiX = obsX(X)
    PsiU = [1]
    PsiH = obsH(X)

    Psi = np.vstack( (PsiX, np.kron(PsiU, PsiH)) )

    return Psi

def rmes(x, Psi, eps=epsilon):
    NkX = obsX()['Nk']

    PsiX = Psi[:NkX].reshape(NkX,1)
    PsiU = [1]
    PsiH = np.vstack( (
        ( anchorMeasure(x) + noise(eps,(Na,1)) )**2,
        [1]
    ) )

    Psin = np.vstack( (PsiX, np.kron(PsiU, PsiH)) )
    return Psin


# helper functions for creating and learning from data
def createData(tList, N0, Nt):
    # generate training data for Kx
    X0 = 10*np.random.rand(Nx,N0) - 5
    xData, uData = generate_data(tList, model, X0,
        control=control, Nu=Nu)

    # formatting training data from xData and uData
    uStack = stack_data(uData, N0, Nu, Nt-1)
    xStack = stack_data(xData[:,:-1], N0, Nx, Nt-1)
    yStack = stack_data(xData[:,1:], N0, Nx, Nt-1)

    XU0 = np.vstack( (X0, np.zeros( (Nu,N0) )) )
    X = np.vstack( (xStack, uStack) )
    Y = np.vstack( (yStack, uStack) )

    return X, Y, XU0

def generateIdealData(sim_time, x0):
    # dimension variables
    Nt = round(sim_time/dt) + 1
    tList = [ [i*dt for i in range(Nt)] ]

    # generate data
    xList, uList = generate_data(tList, model, x0,
        control=control, Nu=Nu)
    return tList, xList, uList


# plotting results helper functions
def stationaryResults(kvar, sim_time, N0n):
    # initial positions
    bounds = 20
    X0n = 2*bounds*np.random.rand(Nx,N0n) - bounds
    U0n = np.zeros( (Nu,N0n) )
    XU0n = np.vstack( (X0n, U0n) )

    # main execution loop
    fig, axs = plt.subplots()
    for i, xu0 in enumerate(XU0n.T):
        x0 = xu0[:Nx,None]
        Psi0 = kvar.obsY.lift( xu0[:,None] )

        tList, xList, PsiList, _, _ = generateTrajectoryData(kvar, sim_time, x0, Psi0)
        animatedResults(tList, xList, PsiList, axs=axs, fig=fig, rush=1, legend=(i==0))

    plotAnchors(fig, axs, radius=1.5)
    axs.set_title('$\\varepsilon=%.2f$' % epsilon)
    return fig, axs

def pathComparisons(kvar, sim_time, x0, Psi0, eList):
    # time sets
    Nt = round(sim_time/dt) + 1
    tList = [ [i*dt for i in range(Nt)] ]

    # plot initialization
    fig, axs = plt.subplots(1,3)

    axs[0].set_title('$x_1$ Path Comparison')
    axs[1].set_title('$x_2$ Path Comparison')

    for ax in axs:
        ax.plot([tList[0][0], tList[0][-1]], [0, 0],
            color='r', linestyle='--', zorder=100, label='Ref')
        ax.set_ylim(0, 1.25)

    for eps in eList:
        xKoop = generateTrajectoryData(kvar, sim_time, x0, Psi0, eps=eps)[1]
        xTrue = generateIdealData(sim_time, x0)[1]

        pathDiff = abs( xKoop - xTrue )
        diffStack = np.hstack( (pathDiff[0], pathDiff[1]) )
        epsAvg = np.average( diffStack )

        axs[0].plot(tList[0], pathDiff[0])
        axs[1].plot(tList[0], pathDiff[1])
        axs[2].plot([tList[0][0], tList[0][-1]], [epsAvg, epsAvg],
            label='$\\varepsilon = %.2f$' % eps)

    axs[2].legend(loc='upper right')
    return fig, axs

def openLoopComparisons(kvar, sim_time, N0n, err=epsilon):
    # dimension variables
    NkXU = obsXU()['Nk']
    Nt = round(sim_time/dt) + 1
    tList = [ [i*dt for i in range(Nt)] ]

    # initial positions
    bounds = 40
    X0n = 2*bounds*np.random.rand(Nx,N0n) - bounds
    U0n = np.zeros( (Nu,N0n) )
    XU0n = np.vstack( (X0n + noise(delta,(Nx,N0n)), U0n) )

    Psi0 = np.empty( (NkXU,N0n) )
    for i, xu in enumerate(XU0n.T):
        Psi0[:,i] = obsXU( xu.reshape(Nx+Nu,1) ).reshape(NkXU,)

    # new operator model equation
    NkX = obsX()['Nk']
    kModel = lambda Psi: kvar.K@rmes(Psi[:Nx,None], Psi)

    xTest, uTest = generate_data(tList, model, X0n,
        control=control, Nu=Nu)
    PsiTest, _ = generate_data(tList, kModel, Psi0)

    # plot results
    xPsi = np.empty( (N0n*Nx, Nt) )
    i = 0;  j = 0
    for k in range(N0n):
        xPsi[i:i+Nx,:] = PsiTest[j:j+Nx,:]
        i += Nx
        j += NkXU
    figComp, axsComp = compare_data(xTest, xPsi, X0n)
    figComp, axsComp = plotAnchors(figComp, axsComp, radius=1.25)
    axsComp.legend()

    axsComp.set_title('$\delta=%.1f, ' % delta + '\\varepsilonilon=%.2f$' % epsilon)
    return figComp, axsComp

def generateTrajectoryData(kvar, sim_time, x0, Psi0, eps=epsilon):
    # dimension variables
    Nt = round(sim_time/dt) + 1
    NkX = obsX()['Nk']
    NkXU = obsXU()['Nk']

    # data list variables
    xList = np.empty( (Nx, Nt) )
    PsiList = np.empty( (NkXU, Nt) )
    uList = np.empty( (Nu, Nt-1) )
    uTrueList = np.empty( (Nu, Nt-1) )

    # initial states for lists
    xList[:,0] = x0[:,0]
    PsiList[:,0] = Psi0[:,0]

    # initialize sim variables and loop
    x = x0
    u = np.zeros( (Nu,1) )
    Psi = Psi0
    for i in range(Nt-1):
        Psi = kvar.K@rmes(x, Psi, eps=eps)
        u = Psi[NkX:]
        x = model(x,u)

        PsiList[:,i+1] = Psi[:,0]
        xList[:,i+1] = x[:,0]

        uList[:,i] = u[:,0]
        uTrueList[:,i] = control(x)[:,0]

    tList = [ [i*dt for i in range(Nt)] ]
    return tList, xList, PsiList, uList, uTrueList

def animatedResults(tList, xList, PsiList, fig=None, axs=None, rush=0, legend=1):
    # if no figure info is given
    if fig is None and axs is None:
        fig, axs = plt.subplots()

    # vehicle variables
    xd = np.zeros( (Nx,1) )
    xvhc = Vehicle2D(xList[:,0,None],
        fig=fig, axs=axs,
        zorder=10,
        radius=0.70, color=mColor,
        tail_length=1000)
    kvhc = Vehicle2D(PsiList[:2,0,None],
        fig=fig, axs=axs,
        zorder=20,
        radius=0.50, color=kColor,
        tail_length=1000)
    xvhc.tail.setLineWidth( 2.00 )
    kvhc.tail.setLineStyle( '--' )
    kvhc.tail.setLineWidth( 1.25 )
    plotAnchors(fig, axs)

    # Axis setup.
    plt.axis( [-20, 20, -20, 20] )
    plt.gca().set_aspect( 'equal', adjustable='box' )
    plt.show( block=0 )

    # propagation function
    NkX = obsX()['Nk']
    Nt = len(tList[0])

    xvhc.draw()
    kvhc.draw()
    for i in range(Nt):
        xvhc.update(xList[:2,i,None])
        kvhc.update(PsiList[:2,i,None])

        if not rush:
            plt.pause( 1e-3 )
            # kvhc.update_title('time: %.3f' % float(i*dt))

    # xvhc.fig.tight_layout()
    return xvhc, kvhc

def trajPlotting(tList, xList, PsiList, uList, uTrueList,
    fig=None, axs=None):
    if fig is None:
        fig, axs = plt.subplots(2,3)

    # iteration list
    Nt = len(tList[0])
    nList = [ [i for i in range(Nt)] ]

    # position comparisons
    axs[0,0].plot(nList[0], xList[0],
        color=mColor, label='Model')
    axs[0,0].plot(nList[0], PsiList[0],
        color=kColor, linestyle='--', label='KFO')
    axs[0,0].set_title('$x$-axis')
    axs[0,0].set_ylabel('$x_1$')

    axs[0,1].plot(nList[0], xList[1],
        color=mColor, label='Model')
    axs[0,1].plot(nList[0], PsiList[1],
        color=kColor, linestyle='--', label='KFO')
    axs[0,1].set_title('$y$-axis')
    axs[0,1].set_ylabel('$x_2$')
    axs[0,1].legend(loc='lower right')

    axs[0,2].plot(nList[0], xList[0]-PsiList[0],
        color=x1Color, label='$x_1$')
    axs[0,2].plot(nList[0], xList[1]-PsiList[1],
        color=x2Color, linestyle='--', label='$x_2$')
    axs[0,2].set_ylim(-2,2)
    axs[0,2].set_title('State Est.')
    axs[0,2].set_ylabel('Error')
    axs[0,2].legend(loc='lower right')

    # input comaprison
    axs[1,0].plot(nList[0][:Nt-1], uTrueList[0],
        color=mColor, label='Model')
    axs[1,0].plot(nList[0][:Nt-1], uList[0],
        color=kColor, linestyle='--', label='KFO')
    axs[1,0].set_ylabel('$u_1$')
    # axs[1,0].set_xlabel('Iteration')

    axs[1,1].plot(nList[0][:Nt-1], uTrueList[1],
        color=mColor, label='Model')
    axs[1,1].plot(nList[0][:Nt-1], uList[1],
        color=kColor, linestyle='--', label='KFO')
    axs[1,1].set_ylabel('$u_2$')
    axs[1,1].set_xlabel('Iteration # (all bottom axes)')

    axs[1,2].plot(nList[0][:Nt-1], uTrueList[0]-uList[0],
        color=x1Color, label='$u_1$')
    axs[1,2].plot(nList[0][:Nt-1], uTrueList[1]-uList[1],
        color=x2Color, linestyle='--', label='$u_2$')
    axs[1,2].set_ylim(-2,2)
    axs[1,2].set_title('Control Est.')
    axs[1,2].set_ylabel('Error')
    # axs[1,2].set_xlabel('Iteration')
    axs[1,2].legend(loc='lower right')

    fig.tight_layout()
    return fig, axs