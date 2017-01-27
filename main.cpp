#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <time.h>
#include "network.h"

//---------------------------------------- ******* ----------------------------------------//
/*
    This a implementation of a neural network in C++ with variable number of layers and neurons.
    With the fact than the programmer is newbie, I will be happy if you teach me something ;)
    For compile this do you will need the Eigen library for linear algebra optimization,
    and a complier with openMP ( -fopenmp -pthread -lpthread ) for multi-threads application.

    The gradient descender method used is the AdaDelta, so you don't need to prove any
    learning parameter. Also I choose to use the convergence rule as stop condition on
    this neural network and I set it to 1E-5. For regularization, you can change the
    define of the LAMBDA value above.
*/
//---------------------------------------- ******* ----------------------------------------//

#define INPUT_FILE "input.txt" // the file when the data of input is allocated
#define OUTPUT_FILE "output.txt" // same
#define COST_FILE "cost.txt" // the file when print the cost history
#define CONVERGENCE 0.00001 // the criterion of convergence
#define EPLISON 1E-8
#define GAMMA 0.95

vector<double> lambda_vec={0.1,0.3,1,3};


using namespace std;
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
typedef Matrix<double, Dynamic, 1> VectorXd;

inline MatrixXd get_input_matrix(const size_t _data_quantity, const size_t _features, const string _input_file)
{
    MatrixXd input_matrix= MatrixXd::Zero(_data_quantity, _features);
    ifstream input_fstream(_input_file);
    double mem=0; string line; size_t row=0, clm=0;
    while(getline(input_fstream,line))
    {
        clm=0;
        istringstream string_to_double(line);
        while(string_to_double>>mem){
            input_matrix(row,clm)=mem;
            clm++;
        }
        VectorXd input_row = input_matrix.row(row);
        double mean = input_row.mean();
        input_row = (input_row.array() - mean).matrix().eval();
        input_row.normalize();
        input_matrix.row(row)=input_row;
        row++;
    }
    return input_matrix;
}

inline MatrixXd get_output_matrix(const size_t _data_quantity, const size_t _outputs)
{
    MatrixXd output_matrix= MatrixXd::Zero(_data_quantity, _outputs);
    ifstream output_stream(OUTPUT_FILE);
    int mem; size_t cont=0;
    while(output_stream>>mem)
    {
        if(mem==10) mem=0;
        output_matrix(cont,mem)=1;
        cont++;
    }
    return output_matrix;
}

int main()
{
    cout << "Enter with the number (size_t) of data with you will train your neural network: \n";
    int k=0; cin>>k;
    const size_t data_quantity=k;
    vector<int> topology;
    cout << " \n Please, enter with the number (size_t) of neurons \n for each layer (no bias) of your neural network: \n";
    while(cin>>k)
        topology.push_back(k);
    MatrixXd input_matrix=get_input_matrix(data_quantity, topology[0],INPUT_FILE);
    MatrixXd output_matrix=get_output_matrix(data_quantity, topology.back());
    vector<int> idex(data_quantity);
    iota(idex.begin(), idex.end(), 0);
    srand((unsigned int) time(0));
    random_shuffle (idex.begin(), idex.end());
    double mem=0, acu=0;
    int i;
    vector<double> acu_mem(lambda_vec.size(),0);
    vector<vector<double>> cost_vec(lambda_vec.size());
    vector<double> acu_test(lambda_vec.size(),0);
    #pragma omp parallel for private(i) schedule(dynamic)
    for(i=0; i<lambda_vec.size();i++){
        Network neural_network(topology);
        vector<double> cost_vec_mem;
        neural_network.train_network(neural_network, CONVERGENCE, lambda_vec[i],
                                     EPLISON, GAMMA, cost_vec_mem, input_matrix, output_matrix , idex);
        acu_mem[i]=neural_network.result_check(input_matrix, output_matrix, idex,0.8,0.6*idex.size()-1);
        cout << "\n The test training with the lambda " << lambda_vec[i] << " was finish with a cs accuracy of " << acu_mem[i] << endl;
        cost_vec[i]=cost_vec_mem;
        neural_network.print_theta(lambda_vec[i]);
        acu_test[i]=neural_network.result_check(input_matrix, output_matrix,idex,1,0.8*idex.size()-1);
    }
    size_t index_max=distance(acu_mem.begin(), max_element(acu_mem.begin(), acu_mem.end()));
    ofstream file(COST_FILE);
    for(auto &it : cost_vec[index_max])
        file << it << endl;
    cout << "\n Process finished. Results: \n For the cross-validation set the lambda as chosen as ";
    cout << lambda_vec[index_max] << "\n per a test with accuracy of " << acu_mem[index_max] << endl;
    cout << "The test set accuracy was " << acu_test[index_max];
    return 0;
}
