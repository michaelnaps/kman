
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
    const int Nt = 1000;

    // Matrix initialization.
    MatrixXd x = MatrixXd::Random(3,1);
    MatrixXd Xdata(Nx,Nt-1);
    MatrixXd Ydata(Nx,Nt-1);

    // Propagate Nt steps of model simulation.
    for (int i(0); i < Nt-1; ++i) {
        Xdata.col(i) = x;
        x = model(x);
        Ydata.col(i) = x;
    }

    // // Check that output is as expected.
    // cout << Xdata << endl;
    // cout << endl << Ydata << endl;

    // Create Regressor variable.
    nap::Regressor regr(Xdata, Ydata);
}
