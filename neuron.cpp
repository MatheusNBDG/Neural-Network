#include <Eigen/Dense>

#include "neuron.h"

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


void Neuron::sigmoid()
{
    act_val[0]=1;
    bool version=1;
    if(z.size()==act_val.size()) version =0;
    #pragma omp parallel for
    for(size_t i=0; i<z.size();i++)
        act_val[i+version]=1/(1+exp(-z[i]));

}

VectorXd Neuron::sigmoid_derivative()
{
    VectorXd z_sd = VectorXd::Zero(z.size());
    #pragma omp parallel for
    for(size_t i=0; i<z.size();i++)
        z_sd[i]=(1/(1+exp(-z[i])))*(1-(1/(1+exp(-z[i]))));
    return z_sd;
}

double Neuron::compute_cost(const VectorXd &_output)
{
    double cost_=0;
    for(size_t i=0;i<act_val.size();i++)
        cost_+=_output[i]*log(act_val[i])+(1-_output[i])*log(1-act_val[i]);
    return -cost_;
}
