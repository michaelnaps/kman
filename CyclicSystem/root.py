import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman')

import numpy as np
import matplotlib.pyplot as plt

from Helpers.Operators import *

import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.path as path

# set global output setting
np.set_printoptions(precision=3, suppress=True);


# hyper paramter(s)
R = 7.5;
dt = 0.01;
Nx = 3;
Nu = 3;
Na = 3;
# aList = np.array( [[10, 12, -15],[10, -7, -13]] );
aList = np.array( [[10, -10, 10], [10, 10, -10]] );

# Na = 5;
# aList = np.array( [[10, 10, -10, -10, -5],[10, -10, -10, 10, -5]] );


# for plotting
aColor = 'indianred';
mColor = 'royalblue';
kColor = 'yellowgreen';
x1Color = 'indianred';
x2Color = 'orange';


# open-loop vehicle class
class Vehicle:
    def __init__(self, Psi0, xd,
                 fig=None, axs=None,
                 buffer_length=10, pause=1e-3,
                 color='k', radius=1,
                 record=0):
        if axs is None and fig is None:
            self.fig, self.axs = plt.subplots();
        else:
            self.fig = fig;
            self.axs = axs;

        # figure scaling
        self.axs.set_xlim(-18,18);
        self.axs.set_ylim(-18,18);
        self.axs.axis('equal');
        self.axs.grid(1);

        # initialize aesthetic parameters
        self.color = color;
        self.body_radius = radius;

        x0 = Psi0[:Nx];
        dList = Psi0[Nx:Nx+Na];
        self.body = patch.Circle(x0, self.body_radius,
            facecolor=self.color, edgecolor='k', zorder=1);
        self.aList = [patch.Circle(x0, np.sqrt(d),
            facecolor="None", edgecolor='k') for d in dList];

        self.axs.add_patch(self.body);
        for a in self.aList:
            self.axs.add_patch(a);

        self.pause = pause;
        self.xd = xd;

        if record:
            plt.show(block=0);
            input("Press enter when ready...");

    def update(self, t, Psi, update_title=1, zorder=1):
        self.body.remove();
        for a in self.aList:
            a.remove();

        dList = Psi[Nx:Nx+Na];
        self.body = patch.Circle(Psi[:Nx,0], self.body_radius,
            facecolor=self.color, edgecolor='k', zorder=zorder);
        self.aList = [patch.Circle(Psi[:Nx], np.sqrt(d),
            facecolor="None", edgecolor='k') for d in dList];

        self.axs.add_patch(self.body);
        for a in self.aList:
            self.axs.add_patch(a);

        if update_title:
            plt.title('iteration: %i' % t);
        plt.pause(self.pause);

        return self;


# model and control functions
def model(x, u):
    A = np.array( [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1] ] );
    B = np.array( [
        [dt, 0, 0],
        [0, dt, 0],
        [0, 0, dt] ] );
    xn = A@x.reshape(Nx,1) + B@u.reshape(Nu,1);
    return xn;

def control(x, v=5):
    u = v*np.array( [
        -np.sin( x[2] ),
        np.cos( x[2] ),
        [1/R]
    ] );
    return u;

def anchorMeasure(x):
    da = np.empty( (Na,1) );
    for i, a in enumerate(aList.T):
        da[i] = np.linalg.norm(x[:2] - a);
    return da;

def randCirc(R=1):
    theta = 2*np.pi*np.random.rand();
    x = [R*np.cos( theta ), R*np.sin( theta )];
    return x;


# observation functions
def obsXU(X=None):
    if X is None:
        meta = {'Nk': 3*Nx+2*Nu+Na+1};
        return meta;

    x = X[:Nx];
    d = anchorMeasure( x );
    u = X[Nx:];

    xx = np.multiply( x,x );
    uu = np.multiply( u,u );
    xu = np.multiply( x,u );

    Psi = np.vstack( (x, d**2, xx, 1, u, uu, xu) );

    return Psi;

def obsX(X=None):
    if X is None:
        meta = {'Nk': 2*Nx+Na+1};
        return meta;

    x = X[:Nx];
    d = anchorMeasure( x );
    xx = np.multiply( x,x );

    Psi = np.vstack( (x, d**2, xx, 1) );

    return Psi;


# plot functions
def plotAnchors(fig, axs, radius=0.5):
    axs.scatter(aList[0][0], aList[0][1],
        color=aColor,
        marker='o', label='Anchor(s)');
    for a in aList.T:
        axs.plot(a[0], a[1]);  # to shape axis around anchors
        circEntity = plt.Circle(a, radius,
            facecolor=aColor, edgecolor='k', zorder=100);
        axs.add_artist( circEntity );
    return fig, axs;

# animate results
def simulateModelWithControl(x0, f, g, N=250):
    # simulate results using vehicle class
    figSim, axsSim = plt.subplots();
    vhc = Vehicle( x0, None, fig=figSim, axs=axsSim,
        record=0, color='yellowgreen', radius=0.5 );
    plotAnchors( figSim,axsSim );
    guideCircle = patch.Circle((0,0), radius=R,
        facecolor='None', edgecolor='r', linestyle='--', zorder=1);
    axsSim.add_patch( guideCircle );

    # Animation loop.
    x = x0;
    for k in range( N ):
        u = g( x );
        x = f( x,u );
        vhc.update( k+1, x, update_title=0 )

    # Return instance of vehicle for plotting.
    return vhc;
