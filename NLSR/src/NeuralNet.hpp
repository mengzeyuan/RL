#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <vector>
#include <cmath>
#include <string>
#include <time.h>
#include <iostream>
using namespace std;

namespace nlsr {
class Layer {
private:
    vector<double> bias;
    vector<vector<double > > weights;
    int input_size;
    int output_size;
    string name;
    vector<double> input_propagate;
    vector<double> output_propagate;
    double learning_rate = 0.01;
public:
    Layer (int input_nodes, int dense_nodes, string name_n)
    :input_size(input_nodes), output_size(dense_nodes), name(name_n)
    {
        // set bias and weights
        /* this->name = name_n;
        this->input_size = input_nodes;
        this->output_size = dense_nodes; */
        srand(time(NULL));
        for (int i = 0; i < input_nodes; ++i) {
            vector<double> weight_layer;
            for (int x = 0; x < dense_nodes; ++x) {
                double positive_random_number = (double)(rand() % 10000 + 1)/10000-0.5;
                weight_layer.push_back(positive_random_number);
            }
            this->weights.push_back(weight_layer);
        }
        for (int i = 0; i < dense_nodes; ++i) {
            double positive_random_number = (double)(rand() % 10000 + 1)/10000-0.5;
            this->bias.push_back(positive_random_number);
        }
    }

    void set_learning_rate (double new_learning_rate) {
        this->learning_rate = new_learning_rate;
    }

    double sigmoid (double x) {
        return 1/(1 + pow(2.71828, -x));
    }

    vector<double> propagate_forward (vector<double> input) {
        this->input_propagate = input;

        if (input.size() != this->input_size) {
            throw invalid_argument(this->name + " input error: Not matching desired input.");
        }

        //vector<double> output = zeros(this->output_size);
        vector<double> output(this->output_size, 0);
        // weights
        for (int i = 0; i < this->input_size; ++i) {
            for (int x = 0; x < this->output_size; ++x) {
                output[x] += input[i] * this->weights[i][x];
            }
        }
        // bias
        for (int i = 0; i < this->output_size; ++i) {
            output[i] += this->bias[i];
        }
        // sigmoid
        for (int i = 0; i < this->output_size; ++i) {
            output[i] = this->sigmoid(output[i]);
        }
        this->output_propagate = output;
        return output;
    }


    vector<double> propagate_backward (vector<double> error, bool is_output) {
        if (error.size() != this->output_size) {
            throw invalid_argument ("(Training " + this->name + ") Doesn't satisfy output_size");
        }

        //vector<double> return_propagate = zeros(this->input_size);
        vector<double> return_propagate(this->input_size, 0);

        // error_backward

        for (int i = 0; i < this->input_size; ++i) {
            for (int x = 0; x < this->output_size; ++x)  {
                double weight = this->weights[i][x];
                double z = this->output_propagate[x];
                double sig_prime = z * (1 - z);
                if (is_output) {
                    return_propagate[i] += (weight * sig_prime * 2 * (z - error[x]));
                }
                else {
                    return_propagate[i] += (weight * sig_prime * 2 * (error[x]));
                }
            }
        }
        // weights
        for (int i = 0; i < this->input_size; ++i) {
            for (int x = 0; x < this->output_size; ++x) {
                double a_l_1 = this->input_propagate[i];
                double z = this->output_propagate[x];
                double sig_prime = z * (1 - z);
                if (is_output) {
                    this->weights[i][x] -= this->learning_rate * (a_l_1 * sig_prime * 2 * (z - error[x]));
                }
                else {
                    this->weights[i][x] -= this->learning_rate * (a_l_1 * sig_prime * 2 * error[x]);
                }
            }
        }

        // bias
        for (int i = 0; i < this->output_size; ++i) {
            double z = this->output_propagate[i];
            double sig_prime = z * (1 - z);
            if (is_output) {
                this->bias[i] -= this->learning_rate * (sig_prime * 2 * (z - error[i]));
            }
            else {
                this->bias[i] -= this->learning_rate * (sig_prime * 2 * error[i]);
            }
        }

        return return_propagate;
    }

    //调试用
    void print_array (vector<double> array, string name = "") {
        if (name != "") {
            cout<<name<<": ";
        }
        for (int i = 0; i < array.size(); ++i) {
            cout<<array[i]<<" ";
        }
        cout<<endl;
    }

    vector<double> zeros (int zeros_count) {
        vector<double> return_array;
        for (int i = 0; i < zeros_count; ++i) {
            return_array.push_back(0);
        }
        return return_array;
    }

    //调试用
    void print_2d_array (vector<vector<double>> array) {
        cout<<"2d ARRAY PRINTING....."<<endl;
        for (int i = 0; i < array.size(); ++i) {
            for (int x = 0; x < array[i].size(); ++x) {
                cout<<array[i][x]<<" ";
            }
            cout<<endl;
        }
        cout<<"2d Array printing ended......"<<endl;
    }
};

class NeuralNetwork {
private:
    vector<Layer> layers;
    int input_layer;
    int output_layer;
public:
    NeuralNetwork () {

    }
    NeuralNetwork (vector<int> layout, double learning_rate);
    vector<double> predict (vector<double> input);
    void backprop (vector<double> x_train, vector<double> y_train, bool did_predict = false);
};
}

#endif