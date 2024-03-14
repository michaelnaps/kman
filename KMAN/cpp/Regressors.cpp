
#include "Regressor.cpp"

namespace nap
{
    DataSet::DataSet(const MatrixXd &Xset)
    {
        X = Xset;
        X0 = Xset.col(0);
        N = Xset.rows();
        M = Xset.cols();
        P = 1;
    }

    DataSet::DataSet(const MatrixXd &Xset, const MatrixXd &X0)
    {
        X = Xset;
        X0 = X0;
        N = Xset.rows();
        M = Xset.cols();
        P = X0.cols();
    }
}