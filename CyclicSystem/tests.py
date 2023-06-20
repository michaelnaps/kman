import numpy as np
from cyclic import *

# hyper parameter(s)
dt = 0.01;
R  = 7.5;

def model(x, u):
    return x + dt*u;

def control(x, v=5.0):
    u = np.array( [
        -v*np.sin( x[2] ),
        v*np.cos( x[2] ),
        [v/R]  # constant rate of change...
    ] );
    return u;

if __name__ == '__main__':
    # simulation time
    T = 10;  Nt = round( T/dt ) + 1;

    # initial position variables
    x0 = np.array( [[R],[0],[0]] );

    # vehicle and guide circle around origin
    fig, axs = plt.subplots();
    vhc = Vehicle( x0, None, fig=fig, axs=axs,
        record=0, color='yellowgreen', radius=0.5 );
    guideCircle = patch.Circle((0,0), radius=R,
        facecolor='None', edgecolor='r', linestyle='--', zorder=1);
    axs.add_patch( guideCircle );

    # simulation loop
    x = x0;
    for i in range( Nt ):
        x = model( x,control(x) );
        vhc.update( i*dt,x );
        # plt.pause(0.01);
