import sys
sys.path.insert(0, '/home/michaelnaps/prog/mpc');

import mpc
import numpy as np


# model is discrete (no ode needed)
def model(x, u, dt):
    dx = [dt*u[0]];
    return dx;

def cost(mvar, xlist, ulist):
    g = xlist[1][0]**2;
    return g;

def obs(x=None):
    pass;


if __name__ == "__main__":
    # state dimension variables
    Nx = 1;
    Nu = 1;

    # create MPC class variable
    params = 1e-3;
    model_type = 'continuous';
    PH = 1;
    mvar = mpc.ModelPredictiveControl('ngd', model, cost, params, Nu,
        num_ssvar=Nx, PH_length=PH, model_type=model_type);

    x0 = [0.5];
    uinit = [0 for i in range(Nu*PH)];
    u = mvar.solve(x0, uinit, output=1)[0];
    print(model(x0, u, params));
