import numpy as np
import matplotlib.pyplot as plt

from Helpers.KoopmanFunctions import *
import Helpers.DataFunctions as data

import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.path as path

# set global output setting
np.set_printoptions(precision=5, suppress=True);


# hyper paramter(s)
eps = 1;
delta = 0.1;
dt = 0.01;
Nx = 2;
Nu = 2;
Na = 3;
aList = np.array( [[10, 10, -10],[10, -10, -10]] );

# vehicle entity for simulation
class Vehicle:
    def __init__(self, x0, xd,
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
        self.axs.set_xlim(-12,12);
        self.axs.set_ylim(-12,12);
        self.axs.axis('equal');
        self.axs.grid(1);

        # initialize buffer (trail)
        self.color = color;
        self.body_radius = radius;

        self.body = patch.Circle(x0[:Nx,0], self.body_radius,
            facecolor=self.color, edgecolor='k', zorder=1);
        self.axs.add_patch(self.body);

        # self.buffer = np.array( [x0[:Nx,0] for i in range(buffer_length)] );
        # self.trail_patch = patch.PathPatch(path.Path(self.buffer),
        #     color=self.color);
        # self.axs.add_patch(self.trail_patch);

        self.pause = pause;
        self.xd = xd;

        if record:
            plt.show(block=0);
            input("Press enter when ready...");

    def update(self, t, x, zorder=1):
        self.body.remove();
        # self.trail_patch.remove();

        # self.buffer[:-1] = self.buffer[1:];
        # self.buffer[-1] = x[:2,0];

        self.body = patch.Circle(x[:Nx,0], self.body_radius,
            facecolor=self.color, edgecolor='k', zorder=zorder);
        self.axs.add_patch(self.body);

        # self.trail_patch = patch.PathPatch(path.Path(self.buffer),
        #     color=self.color, fill=0);
        # self.axs.add_patch(self.trail_patch);

        plt.title('time: %.3f' % t);
        # plt.show(block=0);
        plt.pause(self.pause);

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

def noise(eps, shape):
    return eps*np.random.rand(shape[0], shape[1]) - 2*eps;

def anchorMeasure(x):
    da = np.empty( (Na,1) );
    for i, a in enumerate(aList.T):
        da[i] = (x - a[:,None]).T@(x - a[:,None]);
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
    PsiH = anchorMeasure(x);

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

def learnOperators(X, Y, X0):
    # Ku block diagonal matrix function
    def Mu(kvar):
        m = Nu;
        p = obsX()['Nk'];
        q = 1;
        b = obsH()['Nk'];
        Kblock = np.vstack( (
            np.hstack( (np.eye(p), np.zeros( (p,b*q) )) ),
            np.hstack( (np.zeros( (m,p) ), np.kron( np.eye(q), kvar.K)) )
        ) );
        return Kblock;

    # initialize operator variables and solve
    kuvar = KoopmanOperator(obsH, obsU);
    kxvar = KoopmanOperator(obsXUH, obsXU, M=Mu(kuvar));

    klist = (kxvar, kuvar);
    mlist = (Mu, );
    klist = cascade_edmd(klist, mlist, X, Y, X0);

    # form the cumulative operator
    Kf = klist[0].K @ Mu( klist[1] );
    kvar = KoopmanOperator(obsXUH, obsXU, K=Kf);
    kvar.resError(X, Y, X0);

    return kxvar, kuvar, kvar;

def stationaryResults(N0n):
    NkXU = obsXU()['Nk'];

    # initial positions
    bounds = 15;
    X0n = 2*bounds*np.random.rand(Nx,N0n) - bounds;
    XU0n = np.vstack( (X0n, np.zeros( (Nu,N0n) )) );

    Psi0 = np.empty( (NkXU,N0n) );
    for i, xu in enumerate(XU0n.T):
        Psi0[:,i] = obsXU( xu.reshape(Nx+Nu,1) ).reshape(NkXU,);

    # new operator model equation
    NkX = obsX()['Nk'];
    kModel = lambda Psi: kvar.K@rmes(Psi);
    def rmes(PsiXU):
        x = PsiXU[:Nx].reshape(Nx,1);

        PsiX = PsiXU[:NkX].reshape(NkX,1);
        PsiU = [1];
        PsiH = anchorMeasure(x) + noise(eps,(1,1));

        Psin = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );
        return Psin;

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

def animatedResults(x0):
    # simulation variables
    xd = np.zeros( (Nx,1) );
    xu0 = np.vstack( (x0, np.zeros( (Nu,1) )) );

    # vehicle variables
    xvhc = Vehicle(x0, xd,
        radius=0.5, color='blue', buffer_length=25);
    kvhc = Vehicle(x0, xd,
        fig=xvhc.fig, axs=xvhc.axs,
        radius=0.5, color='yellowgreen', buffer_length=25);
    plotAnchors(xvhc.fig, xvhc.axs);

    # propagation function
    NkX = obsX()['Nk'];
    def rmes(PsiXU):
        x = PsiXU[:Nx];

        PsiX = PsiXU[:NkX];
        PsiU = [1];
        PsiH = anchorMeasure(x) + noise(eps,(1,1));

        Psin = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );
        return kvar.K@Psin;

    # simulation
    x = x0;
    Psi = kvar.obsY(xu0) + kvar.obsY( noise(delta,(Nx+Nu,1)) );

    t = 0;
    while t < 1+dt:
        Psi = rmes(Psi);

        u = Psi[NkX:].reshape(Nu,1);
        x = model(x,u);

        xvhc.update(t, x, zorder=1);
        kvhc.update(t, Psi, zorder=2);
        t += dt;

    return xvhc, kvhc;


# main executable section
if __name__ == "__main__":
    # simulation variables
    T = 1;  Nt = round(T/dt) + 1;
    tList = np.array( [ [i*dt for i in range(Nt)] ] );

    # create data for learning operators
    N0 = 2;
    X, Y, XU0 = createData(tList, N0, Nt);

    kxvar, kuvar, kvar = learnOperators(X, Y, XU0);
    klist = (kxvar, kuvar, kvar);

    for k in klist:
        print(k);


    ans = input("\nStationary or animated sim? [s/a] ");
    if ans == 's':
        # test comparison results
        N0n = 25;
        fig, axs = stationaryResults(N0n);
        plt.show();
    elif ans == 'a':
        # simulation variables
        x0 = 20*np.random.rand(Nx,1)-10;
        xvhc, kvhc = animatedResults(x0);
