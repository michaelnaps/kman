import numpy as np

# Helper data function for generating sets.
def generate_data(tlist, model, X0, control=None, Nu=0):
    Nx = len( X0 );
    N0 = len( X0[0] );
    Nt = len( tlist[0] );

    if control is not None:
        ulist = np.empty( (N0*Nu, Nt-1) );
    else:
        ulist = None;
        Nu = Nx;

    xlist = np.empty( (N0*Nx, Nt) );

    n = 0;
    k = 0;
    for i in range(N0):

        if control is not None:
            u = np.empty( (Nu, Nt-1) );
        x = np.empty( (Nx, Nt) );

        x[:,0] = X0[:,i];

        for t in range(Nt-1):
            if control is not None:
                u[:,t] = control( x[:Nx,t] ).reshape(Nu,);
                x[:,t+1] = model( x[:Nx,t], u[:,t] ).reshape(Nx,);
            else:
                x[:,t+1] = model( x[:Nx,t] ).reshape(Nx,);

        if control is not None:
            ulist[k:k+Nu,:] = u;
            k = k + Nu;

        xlist[n:n+Nx,:] = x;
        n = n + Nx;

    return xlist, ulist;

# Helper data function for stacking sets.
# Equivalent to DataSet.flattenData()
def stack_data(data, N0, Nx, Nt):
    x = np.empty( (Nx, N0*Nt) );

    k = 0;
    t = 0;
    for i in range(N0):
        x[:,t:t+Nt] = data[k:k+Nx,:];
        k += Nx;
        t += Nt;

    return x;

# Class: DataSet
# Assumption: Uniform series/stacked data structure.
class DataSet:
	def __init__(self, X, X0=None):
		self.setNewData(X, X0=X0);

	def setNewData(self, X, X0=None):
		self.X = X;
		self.X0 = X0;

		# If X0 is None, then data is flat.
		if X0 is None:
			self.N, self.P = X.shape;
			self.M = 1;
		# otherwise, data has 3-D shape (M).
		else:
			self.N = len( X0 );			# Number of states
			self.P = len( X[0] );		# Number of time-steps
			self.M = len( X0[0] );		# Number of data sets

	def getDataDimn(self):
		return self.N, self.P, self.M;

	def flattenData(self, suppress=0):
		# Execute flatten if M > 1
		if self.M > 1:
			# Warning.
			if not suppress:
				print( "\nWARING: Flatten function cannot be undone...\n")

			# Initialize flattened data matrix.
			Xflat = np.empty( (self.N, self.M*self.P) );

			# Iterate through and reshape matrix.
			n = 0;
			p = 0;
			for i in range( self.M ):
				Xflat[:,p:p+self.P] = self.X[n:n+self.N,:];
				n += self.N;
				p += self.P;

			# Set internal data to flattened data.
			self.setNewData( Xflat, X0=None );
		# otherwise,
		elif not suppress:
			# print warning.
			print( "\nWARNING: Data already flattened...\n" )

		# Return instance of self.
		return self;

# Class: FiniteDifferenceMethod
# Assumption(s):
#   1). Data set is isolated.
#       - Flat with a single initial point.
#   2). Data has a continuous solution.
class FiniteDifferenceMethod( DataSet ):
    def __init__(self, X, X0=None, h=1e-3, method=-1):
        # Initialize data set.
        DataSet.__init__(self, X, X0=X0);

        # Initialize step-size parameter.
        self.h = h;

        # Initialize FDM method.
        #   1: Highest resolution available (not supported).
        #   2: Two-point resolution.
        #   3: Three-point resolution.
        #   4: Four-point resolution.
        self.method = method;

    @property
    def denominator(self, h=None):
        # Return the denominator coefficient
        #   depending on the method of choice.
        if method is None:
            method = self.method;
        if method == 2:
            return h;
        elif method == 3:
            return 2*h;
        elif method == 4:
            return 12*h;

    def solve(self):
        # Initialize data matrices.
        dX = np.empty( self.X.shape );

        # Primary execution loop.
        for i, x in enumerate( X.T ):
            if i < 2:
                dx = self.forward( x[:,None] );
            elif i > (self.N-2):
                dx = self.backward( x[:,None] );
            else:
                dx = self.central( x[:,None] );
            dX[:,i] = dx[:,0];

        # Return matrices with every point differentiated.
        return dX;

    def forward(self, x):
        return dx;

    def central(self, x):
        pass;

    def backward(self, x):
        pass;