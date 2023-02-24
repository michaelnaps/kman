import sys
sys.path.insert(0, '/home/michaelnaps/prog/mpc');

import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.path as path

import mpc
import Helpers.KoopmanFunctions as kman
import Helpers.DataFunctions as data


# hyper parameter(s)
Nx = 2;
Nu = 1;
dt = 0.01;


# set global output setting
np.set_printoptions(precision=3, suppress=True);

def modelFunc(x, u, _):
    mu = 1;  a = 1;
    x1 = x[0];  x2 = x[1];
    u = u[0];

    xn = [
        x1 + dt*x2,
        x2 + dt*(mu*(1 - x1**2)*x2 - x1 + a*np.sin(u))
    ];

    return xn;

def cost(mvar, xlist, ulist):
    pass;


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
    if x is None:
        meta = {'Nk':3};
        return meta;
    Psi = np.vstack( (x[2], np.sin(x[2]), x[0]**2) );
    return Psi;


if __name__ == "__main__":
    pass;
