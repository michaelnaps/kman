
#include "Regressor.h"

namespace nap
{
    MatrixXd flatten_data(const MatrixXd &X, const int &N)
    {
        // Derive secondary set dimensions.
        const int M = X.cols();
        const int K = X.rows()/N;

        // Initialize flattened set matrix.
        MatrixXd Y(N,M*K);

        // Flatten data.
        int i = 0;
        int j = 0;
        for (int k(0); k < K; ++k) {
            Y.block(0,j,N,M) = X.block(i,0,N,M);
            i += N;
            j += M;
        }

        return Y;
    }

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
        return dmd(TOL);
    }

    MatrixXd Regressor::dmd(const double &EPS)
    {
        // Initialize DMD matrices.
        const double K = Xset.M*Xset.P;
        MatrixXd G = 1/K*(Xset.X*Xset.X.transpose());
        MatrixXd A = 1/K*(Xset.X*Yset.X.transpose());

        // Initialize S^-1 matrix.
        MatrixXd invS = MatrixXd::Zero(Xset.N, Yset.N);

        // Compute SVD on input matrix (for inversion).
        Eigen::JacobiSVD<MatrixXd> svd;
        // TODO: The svd.compute() method is supposed to be deprecated...
        svd.compute(G, Eigen::ComputeFullU | Eigen::ComputeFullV);

        // Invert S matrix.
        MatrixXd S = svd.singularValues();
        for (int i(0); i < Xset.N; ++i) {
            invS(i,i) = 1./S(i);
        }

        // Invert G and calculate operator.
        MatrixXd invG = svd.matrixV()*invS*svd.matrixU().transpose();
        MatrixXd C = A.transpose()*invG;

        cout << "---" << endl;
        const double maxS = S.maxCoeff();
        cout << S << endl << "---" << endl;
        cout << maxS << endl << "---" << endl;
        cout << (S.array() > maxS*EPS) << endl << "---" << endl;

        return C;
    }
}