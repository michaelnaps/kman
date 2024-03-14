
#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using namespace std;

#ifndef KMAN_REGRESSOR
#define KMAN_REGRESSOR

namespace nap
{
    class DataSet
    {
    private:
        // CONSTANT VARIABLES:
        const double TOL = 1e-12;

        // VARIABLES:
        MatrixXd X;     // Data set.
        MatrixXd X0;    // Data set initial conditions.
        int N;          // Length of state space.
        int M;          // Number of data points (per set).
        int P;          // Number of sets.

    public:
        // CONSTRUCTORS:
        DataSet(const MatrixXd &Xdata);
        DataSet(const MatrixXd &Xdata, const MatrixXd &X0data);
    };

    class Regressor
    {
    private:
        // VARIABLES:
        DataSet Xset;
        DataSet Yset;

    protected:
    public:
        // CONSTRUCTORS:
        Regressor(const MatrixXd &Xdata, const MatrixXd &Ydata);
        Regressor(const MatrixXd &Xdata, const MatrixXd &Ydata, const MatrixXd &X0data, const MatrixXd &Y0data);

        // MEMBER FUNCTIONS:
        MatrixXd dmd();
        MatrixXd dmd(const double &EPS);
    };
}

#endif
