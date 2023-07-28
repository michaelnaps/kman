import sys
from os.path import expanduser
sys.path.insert( 0, expanduser( '~' )+'/prog/geom' )

import numpy as np
from GEOM.Vehicle2D import *


# hyper parameter(s)
Nx = 3
dt = 0.005
R  = 7.5

def model(x, u):
    return x + dt*u

def control(x, v=5):
    x = x.reshape(Nx,1)  # reshape x-variable row->col
    u = v*np.array( [
        -x[1]/R,
        x[0]/R,
        [1/R]
    ] )
    return u

if __name__ == '__main__':
    # simulation time
    T = 10;  Nt = round( T/dt ) + 1

    # initial position variables
    x0 = np.array( [[R],[0],[0]] )

    # vehicle and guide circle around origin
    fig, axs = plt.subplots()
    vhc = Vehicle2D( x0[:2], fig=fig, axs=axs, color='yellowgreen',
        radius=0.5, zorder=5 )
    guideCircle = plt.Circle((0,0), radius=R, facecolor='None',
        edgecolor='r', linestyle='--', zorder=1)
    axs.add_patch( guideCircle )
    axs.set_xlim( (-10, 10) )
    axs.axis( 'equal' )

    # simulation loop
    x = x0
    for i in range( Nt ):
        x = model( x,control(x) )
        vhc.update( x[:2] )
        # plt.pause(0.01)
