#ifndef ADADELTA_H_INCLUDED
#define ADADELTA_H_INCLUDED

#include <Eigen/Dense>

//---------------------------------------- ******* ----------------------------------------//
/*
    This a implementation of a neural network in C++ with variable number of layers and neurons.
    With the fact than the programmer is newbie, I will be happy if you teach me something ;)
    For compile this do you will need the Eigen library for linear algebra optimization,
    and a complier with openMP ( -fopenmp -pthread -lpthread ) for multi-threads application.
*/
//---------------------------------------- ******* ----------------------------------------//

using namespace std;
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
typedef Matrix<double, Dynamic, 1> VectorXd;

class AdaDelta
{
public:
    AdaDelta(const size_t _neuron_length, const size_t _next_neuron_length):
        E(MatrixXd::Zero(_next_neuron_length,_neuron_length+1)),
        Esq(MatrixXd::Zero(_next_neuron_length,_neuron_length+1)) {}
    MatrixXd adadelta_algorithm(const MatrixXd &_grad, const double &_gamma, const double &_eplison)
    {
    MatrixXd delta_theta = MatrixXd::Zero(_grad.rows(),_grad.cols());
    #pragma omp parallel for
    for (size_t row=0; row<_grad.rows();row++)
        for (size_t col=0; col!=_grad.cols();col++)
        {
            E(row,col)=_gamma*E(row,col)+(1-_gamma)*_grad(row,col)*_grad(row,col);
            delta_theta(row,col)=-sqrt((Esq(row,col)+_eplison)/(E(row,col)+_eplison))*_grad(row,col);
            Esq(row,col)=_gamma*Esq(row,col)+(1-_gamma)*delta_theta(row,col)*delta_theta(row,col);
        }
    return delta_theta;
    }
private:
    MatrixXd E;
    MatrixXd Esq;
};



#endif // ADADELTA_H_INCLUDED
