import sys
sys.path.insert(0, '/home/michaelnaps/prog/mpc');

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.path as path

import mpc
import Helpers.KoopmanFunctions as kman
import Helpers.DataFunctions as data


# hyper parameter(s)
PH = 2;
xd = [0,2];
Nx = 2;
Nu = 1;
dt = 0.025;


# callback function for runtime
class Parameters:
    def __init__(self, x0,
                 fig=None, axs=None,
                 buffer_length=10, pause=1e-3,
                 color='k'):
        if axs is None and fig is None:
            self.fig, self.axs = plt.subplots();
        else:
            self.fig = fig;
            self.axs = axs;

        self.axs.plot(xd[0], xd[1], color='g', marker='*', markersize=5)
        self.axs.set_xlim(-3, 3);
        self.axs.set_ylim(-4, 4);
        self.fig.tight_layout();

        # initialize buffer (trail)
        self.color = color;

        self.buffer = [x0 for i in range(buffer_length)];
        self.trail_patch = patch.PathPatch(path.Path(self.buffer), color=self.color);

        self.axs.add_patch(self.trail_patch);

        self.pause = pause;

    def update(self, t, x):
        self.trail_patch.remove();

        self.buffer[:-1] = self.buffer[1:];
        self.buffer[-1] = x;

        self.trail_patch = patch.PathPatch(path.Path(self.buffer), fill=0);
        self.axs.add_patch(self.trail_patch);

        plt.show(block=0);
        plt.pause(self.pause);

        return self;

def callback(mvar, T, x, u):
    return mvar.params.update(T, x);


# set global output setting
def model(x, u, _):
    mu = 1;  a = 1;
    x1 = x[0];  x2 = x[1];
    u = u[0];

    xn = [
        x1 + dt*x2,
        x2 + dt*(mu*(1 - x1**2)*x2 - x1 + a*math.sin(u))
    ];

    return xn;

def cost(mvar, xlist, ulist):
    kx = 100;
    kv = 1;
    g = 0;

    for x in xlist:
        g += kx*(xd[0] - x[0])**2 + kv*(xd[1] - x[1])**2;

    return g;


# def obsX(x=None):
#     if x is None:
#         meta = {'Nk':2};
#         return meta;
#     Psi = x[:2];
#     return Psi;
#
# def obsU(x=None):
#     if x is None:
#         meta = {'Nk':2};
#         return meta;
#     Psi = np.vstack( ([1], x[1]) );
#     return Psi;
#
# def obsH(x=None):
#     if x is None:
#         meta = {'Nk':3};
#         return meta;
#     Psi = np.vstack( (x[2], np.sin(x[2]), x[0]**2) );
#     return Psi;


if __name__ == "__main__":
    # test case
    x0 = [1, 0];
    uinit = [0 for i in range(Nu*PH)];

    # create MPC class variable
    params = Parameters(x0, buffer_length=100);
    type = 'discrete';
    mvar = mpc.ModelPredictiveControl('ngd', model, cost, params, Nu,
        num_ssvar=Nx, PH_length=PH, time_step=dt, model_type=type);

    # simulate test case
    sim_time = 10;
    xlist = mvar.sim_root(sim_time, x0, uinit, callback=callback, output=1)[1];
