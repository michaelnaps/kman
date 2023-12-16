import numpy as np

# Helper data function for generating sets.
def generate_data(tlist, model, X0, control=None, Nu=0):
    Nx, N0 = X0.shape
    Nt = tlist.shape[1]

    if control is not None:
        ulist = np.empty( (N0*Nu, Nt-1) )
    else:
        ulist = None
        Nu = 1

    xlist = np.empty( (N0*Nx, Nt) )

    n = 0
    k = 0
    for i in range(N0):

        if control is not None:
            u = np.empty( (Nu, Nt-1) )
        x = np.empty( (Nx, Nt) )

        x[:,0] = X0[:,i]

        for t in range(Nt-1):
            if control is not None:
                u[:,t] = control( x[:Nx,t].reshape(Nx,1) ).reshape(Nu,)
                x[:,t+1] = model( x[:Nx,t].reshape(Nx,1), u[:,t].reshape(Nu,1) ).reshape(Nx,)
            else:
                x[:,t+1] = model( x[:Nx,t].reshape(Nx,1) ).reshape(Nx,)

        if control is not None:
            ulist[k:k+Nu,:] = u
            k = k + Nu

        xlist[n:n+Nx,:] = x
        n = n + Nx

    return xlist, ulist

# Helper data function for stacking sets.
# Equivalent to DataSet.flattenData()
def stack_data(data, N0, Nx, Nt):
    x = np.empty( (Nx, N0*Nt) )

    k = 0
    t = 0
    for i in range(N0):
        x[:,t:t+Nt] = data[k:k+Nx,:]
        k += Nx
        t += Nt

    return x

# Class: DataSet
# Assumption: Uniform series/stacked data structure.
class DataSet:
	def __init__(self, X, X0=None):
		self.setNewData(X, X0=X0)

	def setNewData(self, X, X0=None):
		self.X = X
		self.X0 = X0

		# If X0 is None, then data is flat.
		if X0 is None:
			self.N, self.P = X.shape
			self.M = 1
		# otherwise, data has 3-D shape (M).
		else:
			self.P = X0.shape[1]        # Number of time-steps.
			self.N, self.M = X0.shape   # Number of states/data sets.

	def getDataDimn(self):
		return self.N, self.P, self.M

	def flattenData(self, verbose=0):
		# Execute flatten if M > 1
		if self.M > 1:
			# Warning.
			if verbose:
				print( "\nWARING: Flatten function cannot be undone...\n")

			# Initialize flattened data matrix.
			Xflat = np.empty( (self.N, self.M*self.P) )

			# Iterate through and reshape matrix.
			n = 0
			p = 0
			for i in range( self.M ):
				Xflat[:,p:p+self.P] = self.X[n:n+self.N,:]
				n += self.N
				p += self.P

			# Set internal data to flattened data.
			self.setNewData( Xflat, X0=None )
		# otherwise,
		elif verbose:
			# print warning.
			print( "\nWARNING: Data already flattened...\n" )

		# Return instance of self.
		return self

# NOT CURRENTLY BEING USED/DEVELOPED
# Class: FiniteDifferenceMethod
# Assumption(s):
#   1). Data set is isolated.
#       - Flat with a single initial point.
#   2). Data has a continuous solution.
class FiniteDifferenceMethod( DataSet ):
    def __init__(self, X, X0=None, h=1e-3, method=2):
        # Initialize data set.
        DataSet.__init__(self, X, X0=X0)

        # Initialize step-size parameter.
        self.h = h

        # Initialize FDM method.
        #   1: Highest resolution available (not supported).
        #   2: Two-point resolution.
        #   3: Three-point resolution.
        #   4: Four-point resolution.
        self.method = method

    def denominator(self, method=None):
        # Return the denominator coefficient
        #   depending on the method of choice.
        if method is None:
            method = self.method
        if method == 2:
            return self.h
        elif method == 3:
            return 2*self.h
        elif method == 4:
            return 12*self.h

    def solve(self):
        # Initialize data matrices.
        dX = np.empty( self.X.shape )

        # Primary execution loop.
        n = self.method
        for i in range( self.N ):
            if i < n:
                dx = self.forward( X[:,i:i+n] )
            elif i > (self.N-n):
                dx = self.backward( X[:,i-n:i+n] )
            else:
                dx = self.central( X[:,i-n:i] )
            dX[:,i] = dx[:,0]

        # Return matrices with every point differentiated.
        return dX

    def forward(self, X):
        den = self.denominator()

        if self.method == 2:
            num = -X[:,0] + X[:,1]
        if self.method >= 3:
            num = -3*X[:,0] + 4*X[:,1] - X[:,2]

        return num/den

    def central(self, x):
        den = self.denominator(self.method+1)

        if self.method == 2:
            num = -X[:,0] + X[:,2]
        if self.method >= 3:
            num = X[:,0] - 8*X[:,1] + 8*X[:,3] - X[:,4]

        return num/den

    def backward(self, x):
        den = self.denominator()

        if self.method == 2:
            num = X[:,1] - X[:,0]
        if self.method >= 3:
            num = X[:,0] - 4*X[:,1] + 3*X[:,2]

        return num/den