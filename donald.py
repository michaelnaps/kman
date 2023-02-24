# script for model equation testing
import sys
sys.path.insert(0, '/home/michaelnaps/prog/ode');
sys.path.insert(0, '/home/michaelnaps/prog/mpc');

import mpc
from numpy import random as rd

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.path as path


# hyper parameter(s)
pi = math.pi;
Nx = 3;
Nu = 2;
R = 1/2;  # robot-body radius
dt = 0.01;


# callback function and parameters
class Parameters:
    def __init__(self, x0, xd, PH,
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

def callback(mpc_var, T, x, u):
    xPH = mpc_var.simulate(x, u);
    return mpc_var.params.update(T, x, xPH);


# model functions
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
    kh = 1000;
    kl = 10;
    ku = 1;

    # calculate cost of current input and state
    C = 0;
    k = 0;
    for i, x in enumerate(xlist):
        dx = (x[0] - xd[0])**2 + (x[1] - xd[1])**2;
        do = (x[2] - xd[2])**2;

        # if dx > TOL:
        #     kx = kh;
        #     ko = kl;
        # else:
        #     kx = kl;
        #     ko = kh;

        C += kh*dx;
        C += kl*do;

        if (i != PH):
            C += ku*(ulist[k]**2 + ulist[k+1]**2);
            k += Nu;

    return C;


# observable functions
def obs(x=None):
    if x is None:
        meta = {'Nk':0};
        return meta;
    Psi = np.vstack( (x, 1) );
    return Psi;


if __name__ == "__main__":
    # initialize states
    x0 = [0,0,pi/2];
    xd = [1,1,3*pi/2];

    # create MPC class variable
    PH = 10;
    kl = 2;
    model_type = 'discrete';
    params = Parameters(x0, xd, PH, buffer_length=25, pause=0.1);
    mpc_var = mpc.ModelPredictiveControl('ngd', model, cost, params, Nu,
        num_ssvar=Nx, PH_length=PH, knot_length=kl, time_step=dt,
        max_iter=10, model_type=model_type);
    mpc_var.setAlpha(0.01);

    # solve single time-step
    sim_time = 10;
    uinit = [0 for i in range(Nu*mpc_var.PH)];
    sim_results = mpc_var.sim_root(sim_time, x0, uinit,
        callback=callback, output=1);
    plt.close('all');

    T = sim_results[0];
    xlist = sim_results[1];
    ulist = sim_results[2];
    tlist = sim_results[6];
