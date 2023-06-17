from anchors import *

# hyper parameter(s)
Ntr = 2;
Nsim = 250;

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

        x0 = Psi0[:Nx];
        dList = Psi0[Nx:Nx+Na];
        self.body = patch.Circle(x0, self.body_radius,
            facecolor=self.color, edgecolor='k', zorder=1);
        self.aList = [patch.Circle(x0, np.sqrt(d),
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

# Generate random point on the circumference of a circle.
# Assumption: Center points at origin.
def randCirc(R=1):
    theta = 2*np.pi*np.random.rand();
    x = [R*np.cos(theta), R*np.sin(theta)];
    return x;

# cyclic control function
def cyclicControl(x):
    v = 5;  # constant velocity condition
    u = v*np.array( [
        -x[1]/np.linalg.norm(x),
        x[0]/np.linalg.norm(x)
    ] );
    # th = np.arccos( x[0]/np.linalg.norm( x ) );
    # u = v*np.array( [
    #     -np.sin( th + np.pi/2 ),
    #      np.cos( th + np.pi/2 )
    # ] );
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

# animate results
def animatedResults(kvar):
    # Number of initial points.
    N0 = 1;

    # propagation function
    def prop(PsiX, u):
        x = PsiX[:Nx];
        uu = np.multiply(u,u);
        xu = np.multiply(x,u);

        Psi = np.vstack( (PsiX, u, uu, xu) );
        return kvar.K@Psi;

    x0 = np.array( [ randCirc(R=5) for i in range( N0 )] ).T;
    xu0 = np.vstack( (x0, np.zeros( (Nu,N0) )) );
    Psi0 = np.array( [
        obsX( xu[:,None] )[:,0] for xu in xu0.T
    ] ).T;

    # simulate results using vehicle class
    figSim, axsSim = plt.subplots();
    vhcList = [
        Vehicle( Psi, None, fig=figSim, axs=axsSim,
            record=0, color='yellowgreen', radius=0.5 )
        for Psi in Psi0.T
    ];
    plotAnchors( figSim,axsSim );

    # Animation loop.
    PsiList = Psi0;
    for k in range( Nsim ):
        for i, vhc in enumerate(vhcList):
            u = cyclicControl( PsiList[:Nx,i,None] )
            PsiList[:,i] = prop( PsiList[:,i,None],u )[:,0];
            vhc.update( k+1, PsiList[:,i,None], zorder=10 );

    # Return instance of vehicle for plotting.
    return vhc;

# main execution block
if __name__ == '__main__':
    # simulation data (for training)
    T = 5;  Nt = round(T/dt)+1;
    tList = [[i*dt for i in range(Nt)]];

    # generate data
    N0 = 10;
    X0 = 10*np.random.rand(Nx,N0) - 5;
    # randControl = lambda x: np.random.rand(Nu,1);
    xData, uData = generate_data(tList, model, X0,
        control=cyclicControl, Nu=Nu);

    # stack data appropriately
    uStack = stack_data(uData, N0, Nu, Nt-1);
    xStack = stack_data(xData[:,:-1], N0, Nx, Nt-1);
    yStack = stack_data(xData[:,1:], N0, Nx, Nt-1);

    # create data tuples for training
    XU0 = np.vstack( (X0, np.zeros( (Nu,N0) )) );
    X = np.vstack( (xStack, uStack) );
    Y = np.vstack( (yStack, uStack) );

    # initialize operator
    kvar = KoopmanOperator( obsXU,obsX );
    print( kvar.edmd( X,Y,XU0 ) );

    # animated results
    # ans = input("See results? [a] ");
    # if ans == 'a':
    animatedResults(kvar);
    print("Animation finished...")