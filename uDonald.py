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
dt = 0.001;


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
    ku = 1;

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
def obsX(X=None, mvar=None):
    Ngx = Nx*(PH + 1);
    Ngu = Nu*PH;
    if X is None:
        meta = {'Nk':Ngx+Ngu};
        return meta;

    x = X[:Nx].reshape(Nx,1);
    u = X[-Ngu:].reshape(Ngu,1);

    xList = np.array( mvar.simulate(x, u) ).reshape(Ngx,1);
    gList = np.array( mvar.gradient(x, u) ).reshape(Ngu,1);

    PsiX = np.vstack( (xList, gList) );
    # PsiX = xList;

    return PsiX;

def obsU(X=None, mvar=None):
    Ngx = Nx*(PH + 1);
    Ngu = Nu*PH;
    Ntr = PH + 1;
    if X is None:
        meta = {'Nk':2*Ntr+1};
        return meta;

    x = X[:Nx].reshape(Nx,1);
    u = X[-Ngu:].reshape(Ngu,1);

    xList = np.array( mvar.simulate(x, u) ).reshape(Nx,PH+1);

    xCos = np.cos(xList[2]).reshape(Ntr,1);
    xSin = np.sin(xList[2]).reshape(Ntr,1);
    
    PsiU = np.vstack( (xCos, xSin, [1]) );

    return PsiU;

def obsXU(X=None, mvar=None):
    if X is None:
        meta = {'Nk':obsX()['Nk']+obsU()['Nk']};
        return meta;
    
    PsiX = obsX(X, mvar);
    PsiU = obsU(X, mvar);
    PsiXU = np.vstack( (PsiX, PsiU) );

    return PsiXU;

def obsH(X=None, mvar=None):
    Ngu = Nu*PH;
    if X is None:
        meta = {'Nk':1+Ngu};
        return meta;

    u = X[-Ngu:].reshape(Ngu,1);

    PsiH = np.vstack( ([1], u) );

    return PsiH;

def obsXUH(X=None, mvar=None):
    if X is None:
        meta = {'Nk':obsX()['Nk']+obsU()['Nk']*obsH()['Nk']};
        return meta;

    PsiX = obsX(X, mvar);
    PsiU = obsU(X, mvar);
    PsiH = obsH(X, mvar);

    # kron is not the best operation here
    PsiXUH = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );
    
    return PsiXUH;


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


if __name__ == "__main__":
    # observable dimensions variables
    print("Initializing Variables");
    NkX = obsX()['Nk'];  # for reference
    NkU = obsU()['Nk'];
    NkH = obsH()['Nk'];

    
    # initial position list
    N0 = 1;
    X0 = 2*np.random.rand(Nx,N0) - 1

    
    # initialize states
    x0 = list( X0[:,0].reshape(Nx,) );
    xd = [0,0,pi/2];
    uinit = [0 for i in range(Nu*PH)];

    
    # create MPC class variable
    model_type = 'discrete';
    params = Parameters(x0, xd, buffer_length=25);
    mpc_var = mpc.ModelPredictiveControl('ngd', model, cost, params, Nu,
        num_ssvar=Nx, PH_length=PH, knot_length=kl, time_step=dt,
        max_iter=10, model_type=model_type);
    mpc_var.setAlpha(0.01);


    # simulation time frame
    T = 10;  Nt = round(T/dt) + 1;
    tList = np.array( [[i*dt for i in range(Nt)]] );


    # generate data for training of Ku
    modelFunc = lambda x, u: np.array( model(x, u, None) ).reshape(Nx,1);
    def trainControl(x):  # from MPC class
        umpc = mpc_var.solve(x, uinit)[0];
        return np.array(umpc).reshape(Nu*PH,1);


    # generate and stack data
    print("Generating Training Data");
    xTrain, uTrain = data.generate_data(tList, modelFunc, X0,
        control=trainControl, Nu=Nu*PH);

    uStack = data.stack_data(uTrain, N0, Nu*PH, Nt-1);
    xStack = data.stack_data(xTrain[:,:-1], N0, Nx, Nt-1);
    yStack = data.stack_data(xTrain[:,1:], N0, Nx, Nt-1);


    # solve for K
    X = np.vstack( (xStack, np.zeros( (Nu*PH,N0*(Nt-1)) )) );
    Y = np.vstack( (yStack, uStack) );

    XU0 = np.vstack( (X0, np.zeros( (Nu*PH, N0) )) );
    kvar = kman.KoopmanOperator(obsXUH, obsXU, mpc_var);

    print("Solving for K using EDMD");
    K = kvar.edmd(X, Y, XU0);

    print('K:', K.shape, kvar.err);
    # print(K);

    print('Kx:\n', K[:NkX,:].T);
    # print('Ku:\n', K[NkX:,:].T);


    # simulate results and compare
    print("Generating Comparison Tests");
    koopFunc = lambda Psi: K@rmes(Psi);
    def rmes(Psi):
        PsiX = Psi[:NkX].reshape(NkX,1);
        PsiU = Psi[-NkU:].reshape(NkU,1);
        
        x = Psi[:Nx].reshape(Nx,1);
        u = np.array( uinit ).reshape(Nu*PH,1);
        X = np.vstack( (x, u) );
        PsiH = obsH(X);

        Psin = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );
        return Psin;
    
    x0 = np.array( [[-1],[-1],[pi/2]] );
    xTest, uTest = data.generate_data(tList, modelFunc, x0,
        control=trainControl, Nu=Nu*PH);

    xu0 = np.vstack( (x0, np.array(uinit).reshape(Nu*PH,1)) );
    Psi0 = obsXU(xu0, mpc_var);
    PsiTest = data.generate_data(tList, koopFunc, Psi0)[0];

    plotcomp(xTest, PsiTest);