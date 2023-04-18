from anchors import *

import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.path as path


# parameters for simulation
class Parameters:
    def __init__(self, x0, xd,
                 fig=None, axs=None,
                 buffer_length=10, pause=1e-3,
                 color='k', record=0):
        if axs is None and fig is None:
            self.fig, self.axs = plt.subplots();
        else:
            self.fig = fig;
            self.axs = axs;

        # figure scaling
        self.axs.set_xlim(-12,12);
        self.axs.set_ylim(-12,12);
        self.axs.axis('equal');
        self.axs.grid();
        self.fig.tight_layout();

        # initialize buffer (trail)
        self.color = color;

        self.buffer = np.array( [x0[:Nx,0] for i in range(buffer_length)] );
        self.trail_patch = patch.PathPatch(path.Path(self.buffer),
            color=self.color);
        self.axs.add_patch(self.trail_patch);

        self.pause = pause;
        self.xd = xd;

        if record:
            plt.show(block=0);
            input("Press enter when ready...");

    def update(self, t, x):
        self.trail_patch.remove();

        self.buffer[:-1] = self.buffer[1:];
        self.buffer[-1] = x[:2,0];

        self.trail_patch = patch.PathPatch(path.Path(self.buffer),
            color=self.color, fill=0);
        self.axs.add_patch(self.trail_patch);

        plt.title('time: %.3f' % t);
        plt.show(block=0);
        plt.pause(self.pause);

        return self;


# main executable section
if __name__ == "__main__":
    # simulation variables
    T = 1;  Nt = round(T/dt) + 1;
    tList = np.array( [ [i*dt for i in range(Nt)] ] );

    # create data for learning operators
    N0 = 2;
    X, Y, XU0 = createData(tList, N0, Nt);

    kxvar, kuvar, kvar = learnOperators(X, Y, XU0);
    klist = (kvar, kxvar, kuvar);

    for k in klist:
        print(k);

    # simulation variables
    xd = np.zeros( (Nx,1) );
    x0 = 20*np.random.rand(Nx,1)-10;
    xu0 = np.vstack( (x0, np.zeros( (Nu,1) )) );
    params = Parameters(x0, xd, buffer_length=25);
    plotAnchors(params.fig, params.axs);

    # propagation function
    def rmes(PsiXU):
        x = PsiXU[:Nx];

        PsiX = PsiXU[:p];
        PsiU = [1];
        PsiH = measure(x);

        Psin = np.vstack( (PsiX, np.kron(PsiU, PsiH)) );
        return kvar.K@Psin;

    # simulation
    t = 0;
    Psi = kvar.obsY(xu0);
    while t < 5:
        Psi = rmes(Psi);
        params.update(t, Psi);
        t += dt;
