#include <iostream>
#include "NeuralNet.hpp"
using namespace std;

namespace nlsr {
NeuralNetwork::NeuralNetwork (vector<int> layout, double learning_rate) {
    for (unsigned int i = 0; i < layout.size()-1; ++i) {
        Layer layer_append (layout[i], layout[i+1], "Layer " + to_string(i));
        layer_append.set_learning_rate(learning_rate);
        layers.push_back(layer_append);
    }
    this->input_layer = layout[0];
    this->output_layer = layout[layout.size() - 1];
}

//与miniDNN中network::predict类似
vector<double>
NeuralNetwork::predict (vector<double> input) {
    for (unsigned int i = 0; i < layers.size(); ++i) {
        input = layers[i].propagate_forward(input);
    }
    return input;
}

//与miniDNN中network::backprop类似
void 
NeuralNetwork::backprop (vector<double> x_train, vector<double> y_train, bool did_predict) {
    if (x_train.size() != (unsigned int)this->input_layer) {
        throw invalid_argument("x_train size invalid");
    }
    if (y_train.size() != (unsigned int)this->output_layer) {
        throw invalid_argument("y_train size invalid");
    }
    // predict
    if (!did_predict)
        this->predict(x_train);
    // back propagate
    vector<double> error = y_train;
    for (unsigned int i = this->layers.size() - 1; i > -1; i--) {
        if (i == this->layers.size() - 1) {
            // is output
            vector<double> new_error = this->layers[i].propagate_backward(error, true);
            error = new_error;
        }
        else {
            vector<double> new_error = this->layers[i].propagate_backward(error, false);
            error = new_error;
        }
    }
}

}