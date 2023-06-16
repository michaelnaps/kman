from anchors import *

# hyper parameter(s)
Ntr = 2;

# open-loop vehicle class
class Vehicle:
    def __init__(self, Psi0, xd,
                 fig=None, axs=None,
                 buffer_length=10, pause=1e-3,
                 color='k', radius=1,
                 record=0):
        if axs is None and fig is None:
            self.fig, self.axs = plt.subplots();
        else:
            self.fig = fig;
            self.axs = axs;

        # figure scaling
        self.axs.set_xlim(-12,12);
        self.axs.set_ylim(-12,12);
        self.axs.axis('equal');
        self.axs.grid(1);

        # initialize aesthetic parameters
        self.color = color;
        self.body_radius = radius;

        dList = Psi0[Nx:Nx+Na];
        self.body = patch.Circle(Psi0[:Nx,0], self.body_radius,
            facecolor=self.color, edgecolor='k', zorder=1);
        self.aList = [patch.Circle(Psi0[:Nx], np.sqrt(d),
            facecolor="None", edgecolor='k') for d in dList];

        self.axs.add_patch(self.body);
        for a in self.aList:
            self.axs.add_patch(a);

        self.pause = pause;
        self.xd = xd;

        if record:
            plt.show(block=0);
            input("Press enter when ready...");

    def update(self, t, Psi, zorder=1):
        self.body.remove();
        for a in self.aList:
            a.remove();

        dList = Psi[Nx:Nx+Na];
        self.body = patch.Circle(Psi[:Nx,0], self.body_radius,
            facecolor=self.color, edgecolor='k', zorder=zorder);
        self.aList = [patch.Circle(Psi[:Nx], np.sqrt(d),
            facecolor="None", edgecolor='k') for d in dList];

        self.axs.add_patch(self.body);
        for a in self.aList:
            self.axs.add_patch(a);

        plt.title('iteration: %i' % t);
        plt.pause(self.pause);

        return self;

# cyclic control function
def cyclicControl(x):
    v = 5;  # constant velocity condition
    u = np.array( [
        -v*x[1]/np.linalg.norm(x),
        v*x[0]/np.linalg.norm(x)
    ] );
    return u;

# closed-loop observation functions
# Assumption: ||x|| = 0 is never true.
def obsXU(X=None):
    if X is None:
        meta = {'Nk': 3*Nx+2*Nu+Na+1};
        return meta;

    x = X[:Nx];
    d = anchorMeasure(x);
    u = X[Nx:];

    xx = np.multiply(x,x);
    uu = np.multiply(u,u);
    xu = np.multiply(x,u);

    Psi = np.vstack( (x, d**2, xx, 1, u, uu, xu) );

    return Psi;

def obsX(X=None):
    if X is None:
        meta = {'Nk': 2*Nx+Na+1};
        return meta;

    x = X[:Nx];

    xx = np.multiply(x,x);

    d = np.empty( (Na,1) );
    for i, a in enumerate(aList.T):
        d[i] = (x - a[:,None]).T@(x - a[:,None]);

    Psi = np.vstack( (x, d, xx, 1) );

    return Psi;

def animatedResults(kvar):
    # Initial conditions
    x0 = np.array( [[1],[0]] );
    xu0 = np.vstack( (x0, np.zeros( (Nu,1) )) );
    Psi0 = obsXU( xu0 );

    # simulate results using vehicle class
    vhc = Vehicle(Psi0, None, record=1,
        color='yellowgreen', radius=0.5);
    plotAnchors(vhc.fig, vhc.axs);

    # Simulation loop.
    Ni = 100;
    Psi = Psi0;
    for i in range( Ni ):
        Psi = kvar.K@Psi;
        vhc.update(i+1, Psi, zorder=10);

    # Return final vehicle instance.
    return vhc;

# main execution block
if __name__ == '__main__':
    # simulation data (for training)
    T = 10;  Nt = round(T/dt)+1;
    tList = [[i*dt for i in range(Nt)]];

    # generate data
    N0 = 10;
    X0 = 10*np.random.rand(Nx,N0) - 5;
    xData, uData = generate_data(tList, model, X0,
        control=cyclicControl, Nu=Nu);

    # stack data appropriately
    uStack = stack_data(uData, N0, Nu, Nt-1);
    xStack = stack_data(xData[:,:-1], N0, Nx, Nt-1);
    yStack = stack_data(xData[:,1:], N0, Nx, Nt-1);

    # create data tuples for training
    Nt = len( tList[0] );
    XU0 = np.vstack( (X0, np.zeros( (Nu,N0) )) );
    X = np.vstack( (xStack, np.zeros( (Nu,N0*(Nt-1)) )) );
    Y = np.vstack( (yStack, uStack) );

    # initialize operator
    kvar = KoopmanOperator( obsXU );
    print( kvar.edmd(X, Y, XU0) );

    # animated results
    ans = input("See results? [a] ");
    if ans == 'a':
        animatedResults(kvar);
        print("Animation finished...")