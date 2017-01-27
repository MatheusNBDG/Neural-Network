#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <string>
#include "network.h"
#include "adadelta.h"

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

void Network::input_in_network(const VectorXd &_input)
{
    network[0].act_val[0]=1;
    network[0].act_val.tail(_input.size())=_input;
}


void Network::feed_forward()
{
    for(size_t i=1; i!=network.size(); i++){
        network[i].z=network[i-1].theta*network[i-1].act_val;
        network[i].sigmoid();
    }
}

void Network::back_propagation(const VectorXd &_output, const size_t &_data_quantity)
{
    VectorXd error = network.back().act_val-_output;
    for(size_t i=network.size()-2;i!=-1;i--){
        network[i].grad+=(error*network[i].act_val.transpose())/_data_quantity;
        error=((network[i].theta.transpose()*error).tail(network[i].z.size()).array()*network[i].sigmoid_derivative().array()).matrix().eval();
    }
}

void Network::add_regularization( const double &_lambda,const size_t &_data_quantity)
{
    for(size_t i=0; i!=network.size()-1;i++)
        cost+=_lambda/(2*_data_quantity)*(network[i].theta.squaredNorm());
    #pragma omp parallel for
    for(size_t i=0; i<network.size()-1;i++){
        network[i].grad+=_lambda/_data_quantity*network[i].theta;
        network[i].grad.col(0)-=_lambda/_data_quantity*network[i].theta.col(0);
    }
}

void Network::adadelta_loop(const double &_gamma, const double &_eplison)
{

    #pragma omp parallel for
    for(size_t i=0; i<network.size()-1; i++){
        network[i].theta+=adavec[i].adadelta_algorithm(network[i].grad,_gamma,_eplison);

    }
}

void Network::costf(const VectorXd &_output, const size_t &_data_quantity)
{
    cost+=network.back().compute_cost(_output)/_data_quantity;
}

void Network::train_network(Network &net, const double &_convergence, const double &_lambda,
                   const double &_eplison, const double &_gamma, vector<double> &cost_vec,
                   const MatrixXd &_input, const MatrixXd &_output, const vector<int> &_idex)
{
    double convergence=1, cost_mem=0;
    size_t data_quantity=6*_idex.size()/10;
    while(convergence>_convergence)
    {

        cost=0;
        for(size_t i=0; i!=network.size()-1;i++)
            network[i].grad=MatrixXd::Zero(network[i].grad.rows(),network[i].grad.cols());
        for(size_t idx=0; idx!=data_quantity;idx++)
        {
            input_in_network(_input.row(_idex[idx]));
            feed_forward();
            back_propagation(_output.row(_idex[idx]), data_quantity);
            costf(_output.row(_idex[idx]), data_quantity);
        }
        add_regularization(_lambda, data_quantity);
        adadelta_loop(_gamma,_eplison);
        cost_vec.push_back(return_cost());
        if(return_cost()-cost_mem<=0){
            cout << " \n There is a problem with the convergence of value for lambda equal to " << _lambda;
            cout << ". \n The convergence value is non-positive." << endl;
        }
        convergence=abs(return_cost()-cost_mem);
        cost_mem=return_cost();
    }
}

void Network::print_theta(const double &_lambda)
{
    for (size_t i=0; i!=network.size()-1;i++){
        stringstream i_to_string, lambda_to_string;
        i_to_string << i;
        lambda_to_string << _lambda;
        string theta_file_name = "lambda_"+lambda_to_string.str()+"_theta_"+i_to_string.str()+".txt";
        ofstream file_theta(theta_file_name);
        file_theta << network[i].theta;
        file_theta.close();
    }
}

double Network::result_check(const MatrixXd _input_matrix, const MatrixXd _output_matrix,
                           const vector<int> &_idex, const double &_version, const size_t &_bias)
{
    double correct_rate=0, data_quantity=_idex.size()*_version;
    for(size_t idx=_bias; idx!=data_quantity;idx++){
        input_in_network(_input_matrix.row(_idex[idx]));
        feed_forward();
        MatrixXd::Index max_index;
        network[network.size()-1].act_val.maxCoeff(&max_index);
        if(_output_matrix(_idex[idx],max_index)==1) correct_rate++;
    }
    return correct_rate/(data_quantity-_bias);
}

void Network::load_theta()
{
    for (size_t i=0; i!=network.size()-1;i++){
        stringstream i_to_string;
        i_to_string << i;
        string theta_file_name = "theta"+i_to_string.str()+".txt";
        ifstream file_theta(theta_file_name);
        double mem=0; string line; size_t row=0, clm=0;
        while(getline(file_theta,line))
        {
            clm=0;
            istringstream string_to_double(line);
            while(string_to_double>>mem){
                network[i].theta(row,clm)=mem;
                clm++;
            }
            row++;
        }
    }
}


// ends here
