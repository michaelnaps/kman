
#include "Regressor.h"

namespace nap
{
    DataSet::DataSet(const MatrixXd &Xdata):
        X(Xdata),
        X0(Xdata.col(0)),
        N(Xdata.rows()),
        M(Xdata.cols()),
        P(1) {}

    DataSet::DataSet(const MatrixXd &Xdata, const MatrixXd &X0data):
        X(Xdata),
        X0(X0data),
        N(Xdata.rows()),
        M(Xdata.cols()),
        P(X0data.cols()) {}

    Regressor::Regressor(const MatrixXd &Xdata, const MatrixXd &Ydata):
        Xset(Xdata),
        Yset(Ydata) {}

    Regressor::Regressor(const MatrixXd &Xdata, const MatrixXd &Ydata, const MatrixXd &X0data, const MatrixXd &Y0data):
        Xset(Xdata, X0data),
        Yset(Ydata, Y0data) {}

    MatrixXd Regressor::dmd()
    {

    }

    MatrixXd Regressor::dmd(const double &EPS)
    {

    }
}