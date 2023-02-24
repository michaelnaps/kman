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

        self.axs.set_xlim(-2,2);
        self.axs.set_ylim(-2,2);
        self.axs.axis('equal');
        self.axs.grid();
        self.fig.tight_layout();

        # initialize buffer (trail)
        self.color = color;
        self.width = 0.05;
        self.length = R/2;

        self.buffer = [x0[:2] for i in range(buffer_length)];
        self.trail_patch = patch.PathPatch(path.Path(self.buffer), color=self.color);

        dx1 = self.length*math.cos(x0[2]);
        dx2 = self.length*math.sin(x0[2]);
        self.orientation = patch.Arrow(x0[0], x0[1], dx1, dx2,
                                       width=self.width, color=self.color);

        self.axs.add_patch(self.trail_patch);
        self.axs.add_patch(self.orientation);

        self.pause = pause;
        self.xd = xd;

    def update(self, t, x):
        self.trail_patch.remove();
        self.orientation.remove();

        self.buffer[:-1] = self.buffer[1:];
        self.buffer[-1] = x[:2];

        self.trail_patch = patch.PathPatch(path.Path(self.buffer), fill=0);

        dx1 = self.length*math.cos(x[2]);
        dx2 = self.length*math.sin(x[2]);
        self.orientation = patch.Arrow(x[0], x[1], dx1, dx2,
                                       width=self.width, color=self.color);

        self.axs.add_patch(self.trail_patch);
        self.axs.add_patch(self.orientation);

        plt.show(block=0);
        plt.pause(self.pause);

        return self;


def modelFunc(x, u, _):
    dx = [
        math.cos(x[2])*(u[0] + u[1]),
        math.sin(x[2])*(u[0] + u[1]),
        1/R*(u[0] - u[1])
    ]
    return dx;

def costFunc(mpc_var, xlist, ulist):
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
        do = (math.cos(x[2]) - math.cos(xd[2]))**2 + (math.sin(x[2]) - math.sin(xd[2]))**2;

        if dx > TOL:
            kx = kh;
            ko = kl;
        else:
            kx = kl;
            ko = kh;

        C += kx*dx;
        C += ko*do;

        if (i != PH):
            C += ku*(ulist[k]**2 + ulist[k+1]**2);
            k += Nu;

    return C;

def callbackFunc(mpc_var, T, x, u):
    return mpc_var.params.update(T, x);


if __name__ == "__main__":
    # initialize states
    x0 = [0,0,pi/2];
    xd = [1,1,3*pi/2];

    # create MPC class variable
    PH = 5;
    kl = 2;
    model_type = 'continuous';
    params = Parameters(x0, xd, buffer_length=25);
    mpc_var = mpc.ModelPredictiveControl('ngd', modelFunc, costFunc, params, Nu,
        num_ssvar=Nx, PH_length=PH, knot_length=kl, time_step=dt, model_type=model_type);
    mpc_var.setAlpha(0.01);

    # solve single time-step
    sim_time = 10;
    uinit = [0 for i in range(Nu*mpc_var.PH)];
    sim_results = mpc_var.sim_root(sim_time, x0, uinit,
        callback=callbackFunc, output=1);
    plt.close('all');

    T = sim_results[0];
    xlist = sim_results[1];
    ulist = sim_results[2];
    tlist = sim_results[6];
