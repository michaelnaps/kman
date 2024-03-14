
#include "Regressor.cpp"

MatrixXd model(const MatrixXd &x)
{
    const double dt = 0.01;
    MatrixXd M(3,3);
    M << 1, 2, 0,
        0.5, 1, 0.1,
        0.1, 0, 1;
    return x + dt*M*x;
}

int main()
{
    // Model and simulation dimensions.
    const int Nx = 3;
    const int Nt = 3;
    const int N0 = 2;

    // Matrix initialization.
    MatrixXd x0 = MatrixXd::Random(3,2);
    MatrixXd Xdata(2*Nx,Nt-1);
    MatrixXd Ydata(2*Nx,Nt-1);

    // Output initial condition.
    cout << x0 << endl << "---" << endl;

    // Propagate Nt steps of model simulation.
    MatrixXd x(3,1);
    for (int i(0); i < N0; ++i) {
        x = x0.col(i);
        for (int j(0); j < Nt-1; ++j) {
            Xdata.block<Nx,1>(i*Nx,j) = x;
            x = model(x);
            Ydata.block<Nx,1>(i*Nx,j) = x;
        }
    }

    // Check that output is as expected.
    cout << Xdata << endl << endl;
    cout << Ydata << endl;
    cout << "---" << endl;
    cout << nap::flatten_data( Xdata, Nx ) << endl;

    // Create Regressor variable.
    nap::Regressor regr(Xdata, Ydata);
    cout << regr.dmd() << endl;
}
