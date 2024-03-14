
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
        DataSet(const MatrixXd &X);
        DataSet(const MatrixXd &X, const MatrixXd &X0);
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
    };
}
