#ifndef AGENT_HPP
#define AGENT_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>
#include <float.h>
#include <include/MiniDNN.h>

using namespace std;
using namespace MiniDNN;
//在Eigen中:typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;

namespace nlsr{

class ReplayMemory {
private:
    vector<vector<vector<double>>> mem;
    int capacity;
public:
    vector<double> current_state;
    int action;
    int reward;
    vector<double> next_state;
    bool is_done;
    ReplayMemory () {}
    ReplayMemory (int capacity) {
        this->capacity = capacity;
        srand(time(NULL));
    }
    void store (vector<double> current_state,
		    int action,
		    int reward,
		    vector<double> next_state,
		    bool is_done) {
        if (this->mem.size() == capacity) {
            this->mem.erase(this->mem.begin());
        }

        vector<vector<double>> append_mem;
        append_mem.push_back(current_state);
        append_mem.push_back({(double)action});
        append_mem.push_back({(double)reward});
        append_mem.push_back(next_state);
        append_mem.push_back({(double)is_done});
        this->mem.push_back(append_mem);
    }

    void random () {
        int random_num = rand() % this->mem.size();
        vector<vector<double>> mem_read = this->mem[random_num];
        this->current_state = mem_read[0];
        this->action = (int)mem_read[1][0];
        this->reward = (int)mem_read[2][0];
        this->next_state = mem_read[3];
        this->is_done = (bool)mem_read[4][0];
    }
};

class Agent
{
public:
    int frames = 0;
    Agent() {

    }
    Agent(const vector<int>& layout, const double& lr, const int& mem_capacity, const int& frameReachProb, const int& targetFreqUpdate, const int& batches);
    void transform(const vector<double>& myvec, Matrix& x);
    int max_index(const Matrix& matrix);
    int choose_action(const vector<double>& input);
    void store_mem(const vector<double>& current_state, const int& action, const int& reward, const vector<double>& next_state, const bool& is_done);
    double max_value(const Matrix& matrix);
    void train();

private:
    Network net, target_net;
    Adam opt;
    ReplayMemory mem;
    int frameReachProb;
    int batches;
    int targetFreqUpdate;
    bool is_folder_created = 1;
};

}



#endif