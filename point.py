import numpy as np

import Helpers.KoopmanFunctions as kman
import Helpers.DataFunctions as data


# set global output setting
np.set_printoptions(precision=3, suppress=True);


# hyper paramter(s)
dt = 0.01;
Nx = 4;
Nu = 2;


# model is discrete (no ode needed)
def model(x, u):
    A = np.array( [
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ] );
    B = np.array( [
        [0, 0],
        [0, 0],
        [dt, 0],
        [0, dt]
    ] );

    xn = A@x + B@u;

    return xn;

def control(x):
    C = np.array( [
        [10, 0, 2.5, 0],
        [0, 10, 0, 2.5]
    ] );
    xg = np.array([[0],[0]]);

    u = C*(xg - x.reshape(Nx,1));

    return u;


# observable functions PsiX, PsiU, PsiH
def obs(x=None):
    if x is None:
        meta = {'Nk':obsX()['Nk']+obsU()['Nk']*obsH()['Nk']};
        return meta;
    pass;

def obsX(x=None):
    pass;

def obsU(x=None):
    pass;

def obsH(x=None):
    pass;


if __name__ == "__main__":



