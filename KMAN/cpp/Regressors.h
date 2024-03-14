
#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using namespace std;

namespace nap
{
    class DataSet:
    {
    private:
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

    class Regressor:
    {
    private:
        // VARIABLES:
        DataSet Xset;
        DataSet Yset;

    protected:
    public:
        // CONSTRUCTORS:
        Regressor::Regressor(const MatrixXd &Xdata, const MatrixXd &Ydata);
    };
}
