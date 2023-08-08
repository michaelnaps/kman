from anchors import *

# open-loop vehicle class
class Vehicle:
    def __init__(self, Psi0, xd,
                 fig=None, axs=None,
                 buffer_length=10, pause=1e-3,
                 color='k', radius=1,
                 record=0):
        if axs is None and fig is None:
            self.fig, self.axs = plt.subplots()
        else:
            self.fig = fig
            self.axs = axs

        # initialize aesthetic parameters
        self.color = color
        self.body_radius = radius

        dList = Psi0[Nx:Nx+Na]
        self.body = patch.Circle(Psi0[:Nx,0], self.body_radius,
            facecolor=self.color, edgecolor='k', zorder=1)
        self.aList = [patch.Circle(a, np.sqrt(d),
            facecolor="None", edgecolor='k') for a, d in zip(aList.T, dList)]

        self.axs.add_patch(self.body)
        for a in self.aList:
            self.axs.add_patch(a)

        # Figure settings.
        self.axs.axis('equal')
        self.axs.set_xlim(-15,15)
        self.axs.set_ylim(-15,15)
        self.axs.grid(1)

        # Member variables.
        self.pause = pause
        self.xd = xd

        if record:
            plt.show(block=0)
            input("Press enter when ready...")

    def update(self, t, Psi, zorder=1):
        self.body.remove()
        for a in self.aList:
            a.remove()

        dList = Psi[Nx:Nx+Na]
        self.body = patch.Circle(Psi[:Nx,0], self.body_radius,
            facecolor=self.color, edgecolor='k', zorder=zorder)
        self.aList = [patch.Circle(a, np.sqrt(d),
            facecolor="None", edgecolor='k') for a, d in zip(aList.T, dList)]

        self.axs.add_patch(self.body)
        for a in self.aList:
            self.axs.add_patch(a)

        self.axs.set_xlim(-15,15)
        self.axs.set_ylim(-15,15)

        plt.title('iteration: %i' % t)
        plt.pause(self.pause)

        return self

# closed-loop observation functions
def obsXU(X=None):
    if X is None:
        meta = {'Nk': 3*Nx+2*Nu+Na+1}
        return meta

    x = X[:Nx]
    d = anchorMeasure( x )
    u = X[Nx:]

    xx = np.multiply(x,x)
    uu = np.multiply(u,u)
    xu = np.multiply(x,u)

    Psi = np.vstack( (x, d**2, xx, 1, u, uu, xu) )

    return Psi

def obsX(X=None):
    if X is None:
        meta = {'Nk': 2*Nx+Na+1}
        return meta

    x = X[:Nx]

    xx = np.multiply(x,x)

    d = np.empty( (Na,1) )
    for i, a in enumerate(aList.T):
        d[i] = (x - a[:,None]).T@(x - a[:,None])

    Psi = np.vstack( (x, d, xx, 1) )

    return Psi

# animate results
def animatedResults(kvar):
    # propagation function
    def prop(PsiX, u):
        x = PsiX[:Nx]
        uu = np.multiply(u,u)
        xu = np.multiply(x,u)

        Psi = np.vstack( (PsiX, u, uu, xu) )
        return kvar.K@Psi

    x0 = np.array( [[1],[0]] )
    xu0 = np.vstack( (x0, np.zeros( (Nu,1) )) )
    Psi0 = obsX( xu0 )

    # simulate results using vehicle class
    vhc = Vehicle(Psi0, None, record=1,
        color='yellowgreen', radius=0.5)
    plotAnchors(vhc.fig, vhc.axs)

    A = 5
    Psi = Psi0
    uList = A*np.array( [
         np.cos( np.linspace(0, 2*np.pi, Nt-1) ),
        -np.cos( np.linspace(0, 1.5*np.pi, Nt-1) ) ] )

    for i, u in enumerate(uList.T):
        Psi = prop(Psi, u[:,None])
        # Psi = kvar.K@Psi
        vhc.update(i+1, Psi, zorder=10)

    return vhc

# main execution block
if __name__ == '__main__':
    # simulation data (for training)
    T = 5;  Nt = round(T/dt)+1
    tList = np.array( [[i*dt for i in range(Nt)]] )

    # generate data
    N0 = 1
    X0 = 10*np.random.rand(Nx,N0) - 5
    randControl = lambda x: 5*np.random.rand(Nu,1)-2.5
    xData, uRand = generate_data(tList, model, X0,
        control=randControl, Nu=Nu)

    # stack data appropriately
    uStack = stack_data(uRand, N0, Nu, Nt-1)
    xStack = stack_data(xData[:,:-1], N0, Nx, Nt-1)
    yStack = stack_data(xData[:,1:], N0, Nx, Nt-1)

    # create data tuples for training
    XU0 = np.vstack( (X0, np.zeros( (Nu,N0) )) )
    X = np.vstack( (xStack, uStack) )
    Y = np.vstack( (yStack, uStack) )

    # initialize operator
    kvar = KoopmanOperator( obsXU, obsX )
    print( kvar.edmd(X, Y, XU0) )

    # animated results
    ans = input("See results? [a] ")
    if ans == 'a':
        animatedResults(kvar)
        plt.pause(10)