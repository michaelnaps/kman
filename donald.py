# script for model equation testing
import sys
sys.path.insert(0, '/home/michaelnaps/prog/ode');
sys.path.insert(0, '/home/michaelnaps/prog/mpc');

import mpc
import numpy as np

import Helpers.DataFunctions as data
import Helpers.KoopmanFunctions as kman

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.path as path


# print precision
np.set_printoptions(precision=3, suppress=True);

# hyper parameter(s)
pi = math.pi;
PH = 10;
kl = 2;
Nx = 3;
Nu = 2;
R = 1/2;  # robot-body radius
dt = 0.01;


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
    kh = 150;
    kl = 10;
    ku = 1;

    # calculate cost of current input and state
    C = 0;
    k = 0;
    for i, x in enumerate(xlist):
        dx = (x[0] - xd[0])**2 + (x[1] - xd[1])**2;
        do = (x[2] - xd[2])**2;

        C += kh*dx;
        C += kl*do;

        if (i != PH):
            C += ku*(ulist[k]**2 + ulist[k+1]**2);
            k += Nu;

    return C;


# observable functions
def obs(X=None):
    if X is None:
        meta = {'Nk':obsX()['Nk']+obsU()['Nk']*obsH()['Nk']};
        return meta;

    x = X[:Nx].reshape(Nx,1);

    PsiX = obsX(x);
    PsiU = obsU(x);
    PsiH = obsH(X);

    Psi = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );
    return Psi;

def obsX(x=None):
    if x is None:
        meta = {'Nk':Nx};
        return meta;
    xR = np.array(x).reshape(Nx,1);
    PsiX = xR;
    return PsiX;

def obsU(x=None):
    if x is None:
        Ntrig = 2;
        meta = {'Nk':Ntrig+1};
        return meta;
    PsiU = np.vstack( (np.cos(x[2]), np.sin(x[2]), [1]) );
    return PsiU;

def obsH(X=None):
    if X is None:
        meta = {'Nk':Nu};
        return meta;

    u = X[Nx:].reshape(Nu,1);

    PsiH = u;
    return PsiH;


if __name__ == "__main__":
    # initialize states
    x0 = [-1,-1,3*pi/2];
    xd = [1,1,3*pi/2];
    uinit = [0 for i in range(Nu*PH)];

    # create MPC class variable
    model_type = 'discrete';
    params = Parameters(x0, xd, buffer_length=25);
    mpc_var = mpc.ModelPredictiveControl('ngd', model, cost, params, Nu,
        num_ssvar=Nx, PH_length=PH, knot_length=kl, time_step=dt,
        max_iter=10, model_type=model_type);
    mpc_var.setAlpha(0.01);

    # # solve over 10 [s] time frame
    # sim_time = 10;
    # sim_results = mpc_var.sim_root(sim_time, x0, uinit,
    #     callback=callback, output=1);
    # plt.close('all');

    # # check obs function
    # print(len(obsX(x0)) == obsX()['Nk']);
    # print(len(obsU(uinit)) == obsU()['Nk']);
    # print(len(obs( np.hstack( (x0, uinit) ) )) == obs()['Nk']);

    # model function for training syntax
    modelTrain = lambda x, u: np.array( model(x,u,None) ).reshape(Nx,1);

    # generate initial conditions for training
    N0 = 10;
    X0 = np.random.rand(Nx,N0);

    T = 10;  Nt = round(T/dt)+1;
    tList = [[i*dt for i in range(Nt)]];

    control = lambda x: 10*np.random.rand(Nu,1)-5;
    xTrain, uTrain = data.generate_data(tList, modelTrain, X0, control, Nu);

    # split training data into X and Y sets
    uData = data.stack_data(uTrain, N0, Nu, Nt-1);
    xData = data.stack_data(xTrain[:,:-1], N0, Nx, Nt-1);
    yData = data.stack_data(xTrain[:,1:], N0, Nx, Nt-1);

    X = np.vstack( (xData, uData) );
    Y = np.vstack( (yData, uData) );

    # solve for K
    NkX = obsX()['Nk'];  # for reference
    NkU = obsU()['Nk'];
    NkH = obsH()['Nk'];

    XU0 = np.vstack( (X0, np.zeros( (Nu, N0) )) );
    kxvar = kman.KoopmanOperator(obs);
    Kx = kxvar.edmd(X, Y, XU0);

    print('Kx:', kxvar.err, Kx.shape);
    print(Kx);
