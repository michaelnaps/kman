# script for model equation testing
import sys
sys.path.insert(0, '/home/michaelnaps/prog/mpc');

import mpc
import numpy as np

import Helpers.DataFunctions as data
import Helpers.KoopmanFunctions as kman

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.path as path


# callback function and parameters
class Parameters:
    def __init__(self, x0, xd,
                 fig=None, axs=None,
                 buffer_length=10, pause=1e-3,
                 color='k'):
        if axs is None and fig is None:
            self.fig, self.axs = plt.subplots();
        else:
            self.fig = fig;
            self.axs = axs;

        # figure scaling
        self.axs.set_xlim(-2,2);
        self.axs.set_ylim(-2,2);
        self.axs.axis('equal');
        self.axs.grid();
        self.fig.tight_layout();

        # initialize buffer (trail)
        self.PH = PH;
        self.color = color;
        self.width = 0.10;
        self.length = R/2;

        m1d = self.length*math.cos(xd[2]);
        m2d = self.length*math.sin(xd[2]);
        permanentPatch = patch.Arrow(xd[0], xd[1], m1d, m2d,
            color='g',width=self.width);

        self.buffer = [x0[:2] for i in range(buffer_length)];
        self.trail_patch = patch.PathPatch(path.Path(self.buffer), color=self.color);

        self.prediction = [x0[:2] for i in range(self.PH+1)];
        self.future_patch = patch.PathPatch(path.Path(self.buffer), color='C1');

        m1 = self.length*math.cos(x0[2]);
        m2 = self.length*math.sin(x0[2]);
        self.orientation = patch.Arrow(x0[0], x0[1], m1, m2,
            color=self.color, width=self.width);

        self.axs.add_patch(permanentPatch);
        self.axs.add_patch(self.trail_patch);
        self.axs.add_patch(self.future_patch)
        self.axs.add_patch(self.orientation);

        self.pause = pause;
        self.xd = xd;

        plt.close(self.fig);  # suppress figure from output till update is called

    def update(self, t, x, xPH):
        self.trail_patch.remove();
        self.future_patch.remove();
        self.orientation.remove();

        self.buffer[:-1] = self.buffer[1:];
        self.buffer[-1] = x[:2];

        self.trail_patch = patch.PathPatch(path.Path(self.buffer),
            color=self.color, fill=0);

        self.prediction = [xPH[i][:2] for i in range(self.PH+1)];
        self.future_patch = patch.PathPatch(path.Path(self.prediction),
            color='C1', fill=0);

        dx1 = self.length*math.cos(x[2]);
        dx2 = self.length*math.sin(x[2]);
        self.orientation = patch.Arrow(x[0], x[1], dx1, dx2,
                                       width=self.width, color=self.color);

        self.axs.add_patch(self.trail_patch);
        self.axs.add_patch(self.future_patch);
        self.axs.add_patch(self.orientation);

        plt.show(block=0);
        plt.pause(self.pause);

        return self;

def callback(mvar, T, x, u):
    xPH = mvar.simulate(x, u);
    return mvar.params.update(T, x, xPH);


# print precision
np.set_printoptions(precision=5, suppress=True);


# functions for MPC
def model(x, u, _):
    xn = [
        x[0] + dt*math.cos(x[2])*(u[0] + u[1]),
        x[1] + dt*math.sin(x[2])*(u[0] + u[1]),
        x[2] + dt*1/R*(u[0] - u[1])
    ]
    return xn;

def cost(mpc_var, xlist, ulist):
    # grab class variables
    xd = mpc_var.params.xd;
    Nu = mpc_var.u_num;
    PH = mpc_var.PH;

    # gain parameters
    TOL = 1e-6;
    kx = 1;
    ko = 1;

    # calculate cost of current input and state
    C = 0;
    for i, x in enumerate(xlist):
        gx = (x[0] - xd[0])**2 + (x[1] - xd[1])**2;
        go = (x[2] - xd[2])**2;
        C += kx*gx + ko*go;

    return C;


# plot comparisons
def posTrackingNoControl(tList, kxvar, x0ref, uref):
    # dim variables
    NkX = obsX()['Nk'];
    NkU = obsU()['Nk'];

    # evaluate the behavior of Kx with remeasurement function
    dModel1 = lambda x: np.array( model(x,uref,None) ).reshape(Nx,1);
    kModel1 = lambda Psi: kxvar.K@rmes(Psi);
    def rmes(Psi):
        # tweaks for initial tests
        x = Psi[:Nx].reshape(Nx,1);
        u = uref;
        X = np.vstack( (x,u) );

        PsiX = Psi[:NkX].reshape(NkX,1);
        PsiU = Psi[NkX:].reshape(NkU,1);
        PsiH = obsH(X);

        Psin = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );
        return Psin;

    # test data generation
    Psi0 = obsXU( np.vstack( (x0ref, uref) ) );
    PsiTest = data.generate_data(tList, kModel1, Psi0)[0];
    xTest = data.generate_data(tList, dModel1, x0ref)[0];

    return xTest, PsiTest;

def plotcomp(tList, xTest, PsiTest, save=0):
    # plot test results
    fig, axs = plt.subplots(1,2);

    axs[0].plot(xTest[0], xTest[1], label='Model');
    axs[0].plot(PsiTest[0], PsiTest[1], linestyle='--', label='KCE');

    axs[0].set_xlabel('$x_1$')
    axs[0].set_ylabel('$x_2$')
    axs[0].axis('equal');
    axs[0].grid(1);
    axs[0].legend();

    # evaluate error
    Ne = len( tList[0] ) - 1;
    axs[1].plot([tList[0][0], tList[0][Ne]], [0,0], color='r', linestyle='--');
    axs[1].plot(tList[0][:Ne], PsiTest[0,:Ne]-xTest[0,:Ne], label='$x_1$');
    axs[1].plot(tList[0][:Ne], PsiTest[1,:Ne]-xTest[1,:Ne], label='$x_2$');
    axs[1].plot(tList[0][:Ne], PsiTest[2,:Ne]-xTest[2,:Ne], label='$x_3$');

    axs[1].set_ylabel('Error');
    axs[1].set_xlabel('Time');
    axs[1].set_ylim( (-1,1) );
    axs[1].grid(1);
    axs[1].legend();

    fig.tight_layout();

    # save results
    if save:
        figRes.savefig('/home/michaelnaps/prog/kman/.figures/donald.png', dpi=600);
        figError.savefig('/home/michaelnaps/prog/kman/.figures/donaldError.png', dpi=600);


# hyper parameter(s)
pi = math.pi;
PH = 10;
kl = 2;
Nx = 3;
Nu = 2;
R = 1/2;  # robot-body radius
dt = 0.001;
alpha = 0.01;

# initialize states
x0 = [0,0,0];
xd = [0,0,0];
uinit = [0 for i in range(Nu*PH)];

# create MPC class variable
dt_mpc = 0.01;
model_type = 'discrete';
max_iter = 100;
params = Parameters(x0, xd, buffer_length=25);
mpc_var = mpc.ModelPredictiveControl('ngd', model, cost, params, Nu,
    num_ssvar=Nx, PH_length=PH, knot_length=kl, time_step=dt_mpc,
    max_iter=max_iter, model_type=model_type);
mpc_var.setAlpha(alpha);


# observable functions
def obsX(x=None):
    if x is None:
        meta = {'Nk':Nx};
        return meta;
    PsiX = x;
    return PsiX;

def obsU(x=None):
    if x is None:
        Ntrig = 2;
        meta = {'Nk':Ntrig+1};
        return meta;
    PsiU = np.vstack( (np.cos(x[2]), np.sin(x[2]), [1]) );
    return PsiU;

def obsXU(X=None, mvar=None):
    if X is None:
        meta = {'Nk':obsX()['Nk']+obsU()['Nk']};
        return meta;

    x = X[:Nx].reshape(Nx,1);
    u = X[Nx:].reshape(Nu,1);

    PsiX = obsX(x);
    PsiU = obsU(x);

    PsiXU = np.vstack( (PsiX, PsiU) );
    return PsiXU;

def obsH(X=None, mvar=None):
    if X is None:
        meta = {'Nk':2*Nu*PH};
        return meta;

    u = X[:Nu*PH].reshape(Nu*PH,);
    x = X[Nu*PH:].reshape(Nx,);
    g = np.array( mpc_var.gradient(x, u) );

    PsiH = np.vstack( (u[:,None], g[:,None]) );
    return PsiH;

def obsXUH(X=None, mvar=None):
    if X is None:
        meta = {'Nk':obsX()['Nk']+obsU()['Nk']*obsH()['Nk']};
        return meta;

    x = X[:Nx].reshape(Nx,1);
    u = X[Nx:].reshape(Nu,1);

    PsiX = obsX(x);
    PsiU = obsU(x);
    PsiH = obsH(X, mvar);

    PsiXUH = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );
    return PsiXUH;