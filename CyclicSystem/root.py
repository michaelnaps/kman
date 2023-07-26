import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman')
sys.path.insert(0, expanduser('~')+'/prog/geom')

import numpy as np
import matplotlib.pyplot as plt

from KMAN.Operators import *
from GEOM.Vehicle2D import *
from GEOM.Circle import *

import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.path as path

# set global output setting
np.set_printoptions(precision=3, suppress=True)


# hyper paramter(s)
eps = 0.01
R = 2.5
dt = 0.01
Nx = 3
Ntr = 2
Nu = 3
Na = 3
aList = 1/2*np.array( [[10, -10, 10], [10, 10, -10]] )


# for plotting
aColor = 'indianred'
mColor = 'royalblue'
kColor = 'yellowgreen'
x1Color = 'indianred'
x2Color = 'orange'


# model and control functions
def model(x, u):
    A = np.array( [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1] ] )
    B = np.array( [
        [dt, 0, 0],
        [0, dt, 0],
        [0, 0, dt] ] )
    xn = A@x.reshape(Nx,1) + B@u.reshape(Nu,1)
    return xn

def control(x, v=1):
    x = x.reshape(Nx,1)  # reshape x-variable row->col
    u = v*np.array( [
        -np.sin( x[2] ),
        np.cos( x[2] ),
        [1/R]
    ] )
    return u

def anchorMeasure(x):
    da = np.empty( (Na,1) )
    for i, a in enumerate(aList.T):
        da[i] = np.linalg.norm(x[:2] - a[:,None])
    return da

def randCirc(R=1):
    theta = 2*np.pi*np.random.rand()
    x = [R*np.cos( theta ), R*np.sin( theta )]
    return x

# Noise function.
def noise(alpha, shape):
    return 2*alpha*np.random.rand(shape[0], shape[1]) - alpha

# observation functions
def obsX(X=None):
    if X is None:
        meta = {'Nk': Nx+Ntr+1}
        return meta

    x = X[:Nx]
    xSin = np.cos( x[2] )
    xCos = np.cos( x[2] )

    PsiX = np.vstack( (x, xSin, xCos, [1]) )

    return PsiX

def obsU(X=None):
    if X is None:
        meta = {'Nk': Nu-1}
        return meta

    PsiU = X[Nx:Nx+Nu-1]

    return PsiU

def obsXU(X=None):
    if X is None:
        meta = {'Nk': obsX()['Nk']+obsU()['Nk']}
        return meta

    PsiX = obsX( X )
    PsiU = obsU( X )
    Psi = np.vstack( (PsiX, PsiU) )

    return Psi

def obsH(X=None):
    if X is None:
        meta = {'Nk': Na}
        return meta

    x = X[:Nx]
    d = anchorMeasure( x )
    PsiH = d**2

    return PsiH

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

# add static objects to environment plot
def plotStaticObjects(fig=None, axs=None):
    if fig is None and axs is None:
        fig, axs = plt.subplots()

    plotAnchors( fig, axs )
    guideCircle = patch.Circle((0,0), radius=R,
        facecolor='None', edgecolor='r', linestyle='--', zorder=1)
    axs.add_patch( guideCircle )

    return fig, axs

# animate results
def simulateModelWithControl(x0, F, g=None, N=250, output=0):
    # For 2D simulation.
    N2 = 2

    # Simulate results using vehicle class.
    figSim, axsSim = plotStaticObjects();  axsSim.axis( 'equal' )
    vhc = Vehicle2D( x0[:N2], fig=figSim, axs=axsSim, tail_length=250 )

    # Initialize anchors.
    anchors = [ Circle( a[:,None], d, color='none' ) for a, d in zip( aList.T, anchorMeasure( x0[:2] ) ) ]
    for a in anchors:
        a.draw( fig=figSim, axs=axsSim )

    # Simulation result list.
    Nx = len( x0 )
    xList = np.empty( (Nx,N+1) )

    # Animation loop.
    u = None
    x = x0
    for k in range( N ):
        if g is not None:
            u = g( x )

        x = F( x,u )
        x += noise( eps, (Nx,1) )  # not appropriate place for noise?
        xList[:,k+1] = x[:,0]

        # Update simulation entities.
        vhc.update( x[:N2], pause=0 )
        for a, d in zip( anchors, anchorMeasure( x ) ):
            a.update( radius=d )
        plt.pause( vhc.pause )

        if output:
            print( x.T )

    print("Animation finished...")

    # Return instance of vehicle for plotting.
    return vhc, xList
