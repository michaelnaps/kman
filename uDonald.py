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
kl = 1;
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
    kx = 500;
    ko = 10;
    ku = 0;

    # calculate cost of current input and state
    C = 0;
    k = 0;
    for i, x in enumerate(xlist):
        dx = (x[0] - xd[0])**2 + (x[1] - xd[1])**2;
        do = (x[2] - xd[2])**2;

        C += kx*dx;
        C += ko*do;

        if (i != PH):
            C += ku*(ulist[k]**2 + ulist[k+1]**2);
            k += Nu;

    return C;


# observable functions
def obsH1(X=None, mvar=None):
    Ngx = Nx*(PH + 1);
    Ngu = Nu*PH;
    Ntr = 2;
    if X is None:
        meta = {'Nk':Nx+2*Ngu};
        return meta;

    x = X[:Nx].reshape(Nx,1);
    uinit = X[Nx:].reshape(Ngu,1);

    # xList = np.array( mvar.simulate(x, uinit) ).reshape(Nx,PH+1);
    # uList = uinit.reshape(PH,Nu).T;
    xGrad = np.array( mvar.gradient(x, uinit) ).reshape(Ngu,1);
    # xTrig = np.vstack( (
    #     np.cos(xList[2]).reshape(1,PH+1),
    #     np.sin(xList[2]).reshape(1,PH+1)
    # ) );

    # utr = np.empty( (PH, Ntr*Nu) );
    # for p in range(PH):
    #     x3 = xTrig[:,p].reshape(Ntr,1);
    #     up = uList[:,p].reshape(Nu,1);
    #     utr[p,:] = kman.vec(x3@up.T).reshape(Ntr*Nu,);

    # Psi = np.vstack( (kman.vec(xList), xGrad, kman.vec(utr), uinit) );
    Psi = np.vstack( (x, uinit, xGrad) );
    return Psi;

def obsH2(X=None, mvar=None):
    Ngu = Nu*PH;
    if X is None:
        meta = {'Nk':Nx+Ngu};
        return meta;

    x = X[:Nx].reshape(Nx,1);
    uinit = X[Nx:].reshape(Ngu,1);

    Psi = np.vstack( (x, uinit) );

    return Psi;


# plot comparisons
def plotcomp(xTest, PsiTest, save=0):
    # plot test results
    figRes, axsRes = plt.subplots();

    axsRes.plot(xTest[0], xTest[1], label='Model');
    axsRes.plot(PsiTest[0], PsiTest[1], linestyle='--', label='KCE');

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    axsRes.axis('equal');
    figRes.tight_layout();
    axsRes.legend();
    plt.grid();

    # evaluate error
    Te = 2;  Ne = round(Te/dt) + 1;
    figError, axsError = plt.subplots();

    axsError.plot([tList[0][0], tList[0][Ne]], [0,0], color='r', linestyle='--');
    axsError.plot(tList[0][:Ne], PsiTest[0,:Ne]-xTest[0,:Ne], label='$x_1$');
    axsError.plot(tList[0][:Ne], PsiTest[1,:Ne]-xTest[1,:Ne], label='$x_2$');
    axsError.plot(tList[0][:Ne], PsiTest[2,:Ne]-xTest[2,:Ne], label='$x_3$');

    axsError.set_ylim( (-1,1) );
    axsError.grid();
    axsError.legend();

    # save results
    if save:
        figRes.savefig('/home/michaelnaps/prog/kman/.figures/uDonald.png', dpi=600);
        figError.savefig('/home/michaelnaps/prog/kman/.figures/uDonaldError.png', dpi=600);
    else:
        plt.show();


# brain
if __name__ == "__main__":
    # observable dimensions variables
    print("Initializing Variables");

    
    # initial position list
    N0 = 5;
    X0 = 10*np.random.rand(Nx+Nu*PH,N0) - 5;

    
    # initialize states
    x0 = list( X0[:Nx,0].reshape(Nx,) );
    xd = [0,0,pi/2];
    uinit = [i for i in range(Nu*PH)];

    
    # create MPC class variable
    training_iter = 1;  # max_iter for training purposes
    max_iter = 20;
    model_type = 'discrete';
    params = Parameters(x0, xd, buffer_length=25);
    mpc_var = mpc.ModelPredictiveControl('ngd', model, cost, params, Nu,
        num_ssvar=Nx, PH_length=PH, knot_length=kl, time_step=dt,
        max_iter=training_iter, model_type=model_type);
    mpc_var.setAlpha(0.01);


    # simulation time frame
    # T = 10;  Nt = round(T/dt) + 1;
    iList = np.array( [[i for i in range(max_iter)]] );


    # DATA STRUCTURE IDEA
    #   LENGTH: 10 STEPS (gradient steps)
    #   X0 = np.random.rand(Nx+Nu*PH, N0)
    #   MODEL:
    #       INPUTS: [x0, uinit]
    #       OUTPUT: [x0, unext] (after 1 step in gradient)
    #   CONTROL:
    #       NONE
    def trainModel(X):
        x0 = X[:Nx];
        uinit = X[Nx:];
        un = np.array( mpc_var.solve(x0, uinit)[0] );
        Xn = np.hstack( (x0, un) ).reshape(Nx+Nu*PH,1);
        return Xn;


    # generate and stack data
    print("Generating Training Data");
    xTrain, _ = data.generate_data(iList, trainModel, X0);

    xStack = data.stack_data(xTrain[:,:-1], N0, Nx+Nu*PH, max_iter-1);
    yStack = data.stack_data(xTrain[:,1:], N0, Nx+Nu*PH, max_iter-1);

    X = xStack;  # currently unnecessary (replace x/yStack with X/Y)
    Y = yStack;


    # solve for K
    kvar = kman.KoopmanOperator(obsH1, obsH2, params=mpc_var);

    print("Solving for K using EDMD");
    K = kvar.edmd(X, Y, X0);

    print('K:', K.shape, kvar.err);
    print(K.T);


    # simulate results and compare
    print("Generating Comparison Tests");
    koopFunc = lambda Psi: K@Psi;
    
    x0 = np.random.rand(Nx+Nu*PH,1);
    Psi0 = obsH1(x0, mpc_var);

    x1  = obsH2(trainModel(x0.T[0]), mpc_var);
    Psi1 = K@Psi0;
    
    Nk = obsH2()['Nk'];
    for i in range(Nk):
        print('r:', x1[i], ' K:', Psi1[i], ' err:', x1[i]-Psi1[i]);



    # xTest, uTest = data.generate_data(tList, modelFunc, x0,
    #     control=trainControl, Nu=Nu*PH);

    # PsiTest = data.generate_data(tList, koopFunc, Psi0)[0];

    # plotcomp(xTest, PsiTest);
