#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include<vector>
#include <Eigen/Dense>
#include <cstdlib>
#include <time.h>
#include "adadelta.h"
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

class Network
{
public:
    Network() : network(), topology(), adavec(),cost(0) {}
    Network(vector<int> &_topology):
        network(),topology(_topology),cost(0),adavec()
        {
            for(size_t i=0; i!=topology.size()-1; i++){
                srand((unsigned int) time(0));
                network.push_back(Neuron(topology[i],topology[i+1]));
                adavec.push_back(AdaDelta(topology[i],topology[i+1]));
            }
            network.push_back(Neuron(topology.back()-1,0));
        }
    Network& operator=(const Network &net)
    {
        #pragma omp parallel for
        for(size_t i=0; i<network.size()-1;i++)
            network[i].theta=net.network[i].theta;
    }
    void input_in_network(const VectorXd &_input);
    void feed_forward();
    void back_propagation(const VectorXd &_output, const size_t &_data_quantity);
    void add_regularization( const double &_lambda,const size_t &_data_quantity);
    void adadelta_loop(const double &_gamma, const double &_eplison);
    void costf(const VectorXd &_output, const size_t &_data_quantity);
    double return_cost () { return cost; }
    void train_network (Network &net, const double &_convergence, const double &_lambda,
                        const double &_eplison, const double &_gamma, vector<double> &cost_vec,
                        const MatrixXd &_input, const MatrixXd &_output, const vector<int> &_idex);
    double result_check(const MatrixXd _input_matrix, const MatrixXd _output_matrix,
                      const vector<int> &_index, const double &_version, const size_t &_bias);
    void print_theta(const double &_lambda);
    void load_theta();
private:
    vector<Neuron> network;
    vector<AdaDelta> adavec;
    vector<int> topology;
    double cost;
};


#endif // NETWORK_H_INCLUDED
