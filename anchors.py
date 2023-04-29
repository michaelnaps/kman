import numpy as np
import matplotlib.pyplot as plt

from Helpers.KoopmanFunctions import *
import Helpers.DataFunctions as data

import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.path as path

# set global output setting
np.set_printoptions(precision=3, suppress=True);


# hyper paramter(s)
eps = 0.1;
delta = 1;
dt = 0.01;
Nx = 2;
Nu = 2;
Na = 3;
# aList = np.array( [[10, 12, -15],[10, -7, -13]] );
aList = np.array( [[10, -10, 10], [10, 10, -10]] );

# Na = 5;
# aList = np.array( [[10, 10, -10, -10, -5],[10, -10, -10, 10, -5]] );

# for plotting
mColor = 'royalblue';
kColor = 'yellowgreen';
x1Color = 'indianred';
x2Color = 'orange';


# vehicle entity for simulation
class Vehicle:
    def __init__(self, x0, xd,
                 fig=None, axs=None, zorder=1,
                 buffer_length=10, pause=1e-3,
                 color='k', radius=1,
                 linestyle=None, linewidth=2,
                 record=0):
        if axs is None and fig is None:
            self.fig, self.axs = plt.subplots();
        else:
            self.fig = fig;
            self.axs = axs;

        # figure scaling
        self.axs.set_xlim(-12,12);
        self.axs.set_ylim(-12,12);
        self.axs.axis('equal');
        self.axs.grid(1);

        # initialize buffer (trail)
        self.color = color;
        self.linewidth = linewidth;
        self.linestyle = linestyle;
        self.body_radius = radius;
        self.zorder = zorder;

        self.body = patch.Circle(x0[:Nx,0], self.body_radius,
            facecolor=self.color, edgecolor='k', zorder=self.zorder);
        self.axs.add_patch(self.body);

        self.buffer = [x0[:Nx,0] for i in range(buffer_length)];
        self.trail_patch = patch.PathPatch(path.Path(self.buffer),
            color=self.color, linewidth=self.linewidth, linestyle=self.linestyle,
            zorder=self.zorder);
        self.axs.add_patch(self.trail_patch);

        self.pause = pause;
        self.xd = xd;

        if record:
            plt.show(block=0);
            input("Press enter when ready...");

    def update(self):
        self.body.remove();
        self.trail_patch.remove();

        self.body = patch.Circle(self.buffer[-1], self.body_radius,
            facecolor=self.color, edgecolor='k', zorder=self.zorder);
        self.axs.add_patch(self.body);

        self.trail_patch = patch.PathPatch(path.Path(self.buffer),
            color=self.color, linewidth=self.linewidth, linestyle=self.linestyle,
            fill=0, zorder=self.zorder);
        self.axs.add_patch(self.trail_patch);

        # plt.show(block=0);
        plt.pause(self.pause);

        return self;

    def update_buffer(self, x):
        self.buffer[:-1] = self.buffer[1:];
        self.buffer[-1] = x[:2,0];
        return self;

    def update_title(self, string):
        plt.title(string);
        return self;


# plot functions
def plotAnchors(fig, axs):
    for a in aList.T:
        axs.plot(a[0], a[1]);  # to shape axis around anchors
        circEntity = plt.Circle(a, 0.5, facecolor='indianred', edgecolor='k');
        axs.add_artist( circEntity );
    # axs.axis('equal');
    return fig, axs;


# model and control functions
def model(x, u):
    A = np.array( [
        [1, 0],
        [0, 1] ] );
    B = np.array( [
        [dt, 0],
        [0, dt] ] );

    xn = A@x.reshape(Nx,1) + B@u.reshape(Nu,1);

    return xn;

def control(x):
    C = np.array( [
        [1, 0],
        [0, 1]
    ] );
    xg = np.zeros( (Nx,1) );

    u = C@(xg - x.reshape(Nx,1));

    return u;

def noise(alpha, shape):
    return 2*alpha*np.random.rand(shape[0], shape[1]) - alpha;

def anchorMeasure(x):
    da = np.empty( (Na,1) );
    for i, a in enumerate(aList.T):
        da[i] = np.linalg.norm(x - a[:,None]);
    return da;


# observable functions PsiX, PsiU, PsiH
def obsX(X=None):
    if X is None:
        meta = {'Nk':Nx};
        return meta;
    PsiX = X[:Nx].reshape(Nx,1);
    return PsiX;

def obsU(X=None):
    if X is None:
        meta = {'Nk':Nu};
        return meta;

    PsiU = X[Nx:].reshape(Nu,1);
    return PsiU;

def obsXU(X=None):
    if X is None:
        meta = {'Nk':obsX()['Nk']+obsU()['Nk']};
        return meta;
    PsiX = obsX(X);
    PsiU = obsU(X);
    PsiXU = np.vstack( (PsiX, PsiU) );
    return PsiXU;

def obsH(X=None):
    if X is None:
        meta = {'Nk':Na};
        return meta;

    x = X[:Nx].reshape(Nx,1);
    PsiH = anchorMeasure(x)**2;

    return PsiH;

def obsXUH(X=None):
    if X is None:
        meta = {'Nk':obsX()['Nk']+1*obsH()['Nk']}
        return meta;

    PsiX = obsX(X);
    PsiU = [1];
    PsiH = obsH(X);

    Psi = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );

    return Psi;

def rmes(x, Psi):
    NkX = obsX()['Nk'];

    PsiX = Psi[:NkX].reshape(NkX,1);
    PsiU = [1];
    PsiH = ( anchorMeasure(x) + noise(eps,(Na,1)) )**2;

    Psin = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );
    return Psin;


# helper functions for creating and learning from data
def createData(tList, N0, Nt):
    # generate training data for Kx
    X0 = 10*np.random.rand(Nx,N0) - 5;
    xData, uData = data.generate_data(tList, model, X0,
        control=control, Nu=Nu);

    # formatting training data from xData and uData
    uStack = data.stack_data(uData, N0, Nu, Nt-1);
    xStack = data.stack_data(xData[:,:-1], N0, Nx, Nt-1);
    yStack = data.stack_data(xData[:,1:], N0, Nx, Nt-1);

    XU0 = np.vstack( (X0, np.zeros( (Nu,N0) )) );
    X = np.vstack( (xStack, uStack) );
    Y = np.vstack( (yStack, uStack) );

    return X, Y, XU0;

def stationaryResults(kvar, sim_time, N0n):
    # dimension variables
    NkXU = obsXU()['Nk'];
    Nt = round(sim_time/dt) + 1;
    tList = [ [i*dt for i in range(Nt)] ];

    # initial positions
    bounds = 40;
    X0n = 2*bounds*np.random.rand(Nx,N0n) - bounds;
    XU0n = np.vstack( (X0n, np.zeros( (Nu,N0n) )) );

    Psi0 = np.empty( (NkXU,N0n) );
    for i, xu in enumerate(XU0n.T):
        Psi0[:,i] = obsXU( xu.reshape(Nx+Nu,1) ).reshape(NkXU,);

    # new operator model equation
    NkX = obsX()['Nk'];
    kModel = lambda Psi: kvar.K@rmes(Psi[:Nx,None], Psi);

    xTest, uTest = data.generate_data(tList, model, X0n,
        control=control, Nu=Nu);
    PsiTest, _ = data.generate_data(tList, kModel, Psi0);

    # plot results
    xPsi = np.empty( (N0n*Nx, Nt) );
    i = 0;  j = 0;
    for k in range(N0n):
        xPsi[i:i+Nx,:] = PsiTest[j:j+Nx,:];
        i += Nx;
        j += NkXU;
    figComp, axsComp = data.compare_data(xTest, xPsi, X0n);
    fogComp, axsComp = plotAnchors(figComp, axsComp);

    return figComp, axsComp;

def animatedResults(kvar, T, x0, rush=0):
    # simulation variables
    xd = np.zeros( (Nx,1) );
    xu0 = np.vstack( (x0, np.zeros( (Nu,1) )) );

    # simulation variables
    x = x0;
    Psi = kvar.obsY(xu0) + kvar.obsY( noise(delta,(Nx+Nu,1)) );

    # vehicle variables
    xvhc = Vehicle(x, xd,
        zorder=10,
        radius=0.70, color=mColor,
        linewidth=2,
        buffer_length=10000);
    kvhc = Vehicle(Psi, xd,
        fig=xvhc.fig, axs=xvhc.axs,
        zorder=20,
        radius=0.50, color=kColor,
        linewidth=2, linestyle='--',
        buffer_length=10000);
    plotAnchors(xvhc.fig, xvhc.axs);

    # propagation function
    NkX = obsX()['Nk'];
    Nt = round(T/dt) + 1;

    for i in range(Nt):
        Psi = kvar.K@rmes(x, Psi);

        u = Psi[NkX:].reshape(Nu,1);
        x = model(x,u);

        xvhc.update_buffer(x);
        kvhc.update_buffer(Psi);

        if not rush:
            xvhc.update();
            kvhc.update();
            # kvhc.update_title('time: %.3f' % float(i*dt));

    if rush:
        xvhc.update();
        kvhc.update();

    xvhc.fig.tight_layout();
    return xvhc, kvhc;


# plot position, input and error trajectories for isolated path
def trajSimulation(kvar, tList, x0):
    # dimension variables
    Nt = len( tList[0] );
    NkX = obsX()['Nk'];
    NkXU = kvar.obsY()['Nk'];

    # initialize Psi0
    u0 = np.zeros( (Nu,1) );
    xu0 = np.vstack( (x0, u0) );
    xu0noise = np.vstack( (noise(delta, (Nx,1)), u0) );
    Psi0 = kvar.obsY( xu0+xu0noise );

    # data list variables
    xList = np.empty( (Nx, Nt) );
    PsiList = np.empty( (NkXU, Nt) );
    uList = np.empty( (Nu, Nt-1) );
    uTrueList = np.empty( (Nu, Nt-1) );

    # initial states for lists
    xList[:,0] = x0[:,0];
    PsiList[:,0] = Psi0[:,0];

    # initialize sim variables and loop
    x = x0;
    u = np.zeros( (Nu,1) );
    Psi = Psi0;
    for i in range(Nt-1):
        Psi = kvar.K@rmes(x, Psi);
        u = Psi[NkX:];
        x = model(x,u);

        PsiList[:,i+1] = Psi[:,0];
        xList[:,i+1] = x[:,0];

        uList[:,i] = u[:,0];
        uTrueList[:,i] = control(x)[:,0];

    return xList, PsiList, uList, uTrueList;

def trajPlotting(kvar, sim_time, x0,
    fig=None, axs=None):
    if fig is None:
        fig, axs = plt.subplots(2,3)

    # trajectory simulation
    Nt = round(sim_time/dt) + 1;
    nList = [ [i for i in range(Nt)] ];
    xList, PsiList, uList, uTrueList = trajSimulation(kvar, nList, x0);

    # position comparisons
    axs[0,0].plot(nList[0], xList[0],
        color=mColor, label='Model');
    axs[0,0].plot(nList[0], PsiList[0],
        color=kColor, linestyle='--', label='KCE');
    axs[0,0].set_ylabel('$x_1$')

    axs[0,1].plot(nList[0], xList[1],
        color=mColor, label='Model');
    axs[0,1].plot(nList[0], PsiList[1],
        color=kColor, linestyle='--', label='KCE');
    axs[0,1].set_ylabel('$x_2$')
    axs[0,1].legend();

    axs[0,2].plot(nList[0], xList[0]-PsiList[0],
        color=x1Color, label='$x_1$');
    axs[0,2].plot(nList[0], xList[1]-PsiList[1],
        color=x2Color, linestyle='--', label='$x_2$');
    axs[0,2].set_ylabel('Error');
    axs[0,2].legend();

    # input comaprison
    axs[1,0].plot(nList[0][:Nt-1], uTrueList[0],
        color=mColor, label='Model');
    axs[1,0].plot(nList[0][:Nt-1], uList[0],
        color=kColor, linestyle='--', label='KCE');
    axs[1,0].set_ylabel('$u_1$')
    # axs[1,0].set_xlabel('Iteration')

    axs[1,1].plot(nList[0][:Nt-1], uTrueList[1],
        color=mColor, label='Model');
    axs[1,1].plot(nList[0][:Nt-1], uList[1],
        color=kColor, linestyle='--', label='KCE');
    axs[1,1].set_ylabel('$u_2$')
    axs[1,1].set_xlabel('Iteration #')

    axs[1,2].plot(nList[0][:Nt-1], uTrueList[0]-uList[0],
        color=x1Color, label='$u_1$');
    axs[1,2].plot(nList[0][:Nt-1], uTrueList[1]-uList[1],
        color=x2Color, linestyle='--', label='$u_2$');
    axs[1,2].set_ylabel('Error');
    # axs[1,2].set_xlabel('Iteration')
    axs[1,2].legend();

    fig.tight_layout();
    return fig, axs;