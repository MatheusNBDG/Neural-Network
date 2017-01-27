#ifndef NEURON_H_INCLUDED
#define NEURON_H_INCLUDED

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

class Network;

class Neuron
{
public:
    Neuron();
    Neuron(const size_t _neuron_length,const size_t _next_neuron_length):
        z(VectorXd::Zero(_neuron_length)),
        act_val(VectorXd::Zero(_neuron_length+1)),
        theta(MatrixXd::Random(_next_neuron_length,_neuron_length+1)),
        grad(MatrixXd::Zero(_next_neuron_length,_neuron_length+1))
        {
            theta*=2*sqrt(6)/sqrt(_neuron_length+_next_neuron_length);
        }
    friend Network;
    void sigmoid();
    VectorXd sigmoid_derivative();
    double compute_cost(const VectorXd &_output);
private:
    VectorXd z, act_val; // input on the layer is z and the activation value of neuron is act_vl leight
    MatrixXd theta, grad;
};

#endif // NEURON_H_INCLUDED
