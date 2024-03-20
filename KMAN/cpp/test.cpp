
#include "Regressor.cpp"

MatrixXd model(const MatrixXd &x, const MatrixXd &M)
{
    return x + M*x;
}

int main()
{
    // Model and simulation dimensions.
    const int Nx = 4;
    const int Nt = 50;
    const int N0 = 3;

    // Propagation matrices.
    const double dt = 0.1;
    const MatrixXd M = MatrixXd::Random(Nx,Nx);

    // Matrix initialization.
    MatrixXd x0 = MatrixXd::Random(Nx,N0);
    MatrixXd Xdata(N0*Nx,Nt-1);
    MatrixXd Ydata(N0*Nx,Nt-1);

    // Propagate Nt steps of model simulation.
    MatrixXd x(Nx,1);
    for (int i(0); i < N0; ++i) {
        x = x0.col(i);
        for (int j(0); j < Nt-1; ++j) {
            Xdata.block<Nx,1>(i*Nx,j) = x;
            x = model(x, dt*M);
            Ydata.block<Nx,1>(i*Nx,j) = x;
        }
    }

    // // Check that output is as expected.
    // cout << Xdata << endl << endl;
    // cout << Ydata << endl;
    // cout << "---" << endl;
    // cout << nap::flatten_data( Xdata, Nx ) << endl;

    // Flatten data matrices for multiple initial conditions.
    MatrixXd Xflat = nap::flatten_data( Xdata, Nx );
    MatrixXd Yflat = nap::flatten_data( Ydata, Nx );

    // Create Regressor variable.
    nap::Regressor regr(Xflat, Yflat);

    // Print fitted operator to actual.
    cout << MatrixXd::Identity(Nx,Nx) + dt*M << endl;
    cout << "---" << endl;
    cout << regr.dmd() << endl;
}
