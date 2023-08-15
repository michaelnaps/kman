import numpy as np


# Duffing model function.
def model2(x, c=[5, 0.2, 1], dt=0.001):
    dx = np.array( [
        x[1],
        c[0]*x[0] - c[1]*x[0]**3 - c[2]*x[1]
    ] )
    return x + dt*dx

def model3(x, c=[ 1, 1, 0.07, 0.20, 1.10 ], dt=0.001):
    dx = np.array( [
        x[1],
        c[0]*x[0] - c[1]*x[0]**3 - c[2]*x[1] - c[3]*np.cos( c[4]*x[2] ),
        1  # state variable for tracking time.
    ] )
    return x + dt*dx