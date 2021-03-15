#include "agent.hpp"

namespace nlsr{

Agent::Agent(const vector<int>& layout, const double& lr, const int& mem_capacity, const int& frameReachProb, const int& targetFreqUpdate, const int& batches)
:frameReachProb(frameReachProb), batches(batches), targetFreqUpdate(targetFreqUpdate)
{
    Layer* layer1 = new FullyConnected<ReLU>(layout[0], layout[1]);
    Layer* layer2 = new FullyConnected<ReLU>(layout[1], layout[2]);
    Layer* layer3 = new FullyConnected<Identity>(layout[2], layout[3]);
    net.add_layer(layer1);
    net.add_layer(layer2);
    net.add_layer(layer3);
    net.set_output(new RegressionMSE());
    opt.m_lrate = lr;
    net.init(0, 0.01, 000);
    net.export_net("./NetFolder/", "NetFile_init", is_folder_created);
    target_net.read_net("./NetFolder/", "NetFile_init");
    this->mem = ReplayMemory(mem_capacity);
    srand(time(NULL));
}

void Agent::transform(const vector<double>& myvec, Matrix& x) {
    for(unsigned int i=0; i<myvec.size(); ++i) {
        x(i,0) = myvec[i];
    }
}

int Agent::max_index(const Matrix& matrix) {
    double max = DBL_MIN;
    int index;
    for(unsigned int i=0; i<matrix.rows(); ++i) {
        if(matrix(i,0)>max){
            max = matrix(i,0);
            index = i;
        }
    }
    return index;
}

int Agent::choose_action(const vector<double>& input) {
    this->frames++;
    double probability;
    if (frames <= frameReachProb) {
        probability = (-0.9 / double(frameReachProb)) * frames + 1;
    } else {
        probability = 0.1;
    }
    bool isRandom = (rand() % 100) < (probability * 100);
    int action;
    if (isRandom) {
        action = rand() % 4;
    }
    else {
        Matrix input_x(input.size(),1);
        transform(input, input_x);
        Matrix last_prediction = this->net.predict(input_x);
        action = max_index(last_prediction);
    }
    return action;
}

void Agent::store_mem (const vector<double>& current_state, const int& action, const int& reward, const vector<double>& next_state, const bool& is_done) {
    mem.store(current_state, action, reward, next_state, is_done);
}

double Agent::max_value(const Matrix& matrix) {
    double max = DBL_MIN;
    for(unsigned int i=0; i<matrix.rows(); ++i) {
        if(matrix(i,0)>max){
            max = matrix(i,0);
        }
    }
    return max;
}

void Agent::train () {
    // sample minibatch
    for (int i = 0; i < batches; ++i) {
        mem.random();
        vector<double> current_state = mem.current_state;
        Matrix current_state_matrix(2,1);
        transform(current_state, current_state_matrix);
        int action = mem.action;
        int reward = mem.reward;
        vector<double> next_state = mem.next_state;
        Matrix next_state_matrix(2,1);
        transform(next_state, next_state_matrix);
        bool is_done = mem.is_done;
        // train
        double y;
        if (is_done) {
            y = reward;
        } else {
            //y = reward + (0.99 * max(this->target_net.predict(next_state)));
            y = reward + 0.99*max_value(target_net.predict(next_state_matrix));
        }
        Matrix eval = this->net.predict(current_state_matrix);//eval是一个列矩阵
        Matrix target = eval;
        target(action,0) = y;
        this->net.backprop(current_state_matrix, target);
        net.update(opt);
    }
    if (frames % this->targetFreqUpdate == 0) {
        static int i=0;
        i++;
        string fileName = "NetFile_update"+std::to_string(i);
        net.export_net("./NetFolder/", fileName, is_folder_created);
        target_net.read_net("./NetFolder/", fileName);
    }
}

    /* Agent::Agent(){

    }

    void Agent::check_state_exist(const Observation& observation){
        if(q_table.find(observation) == q_table.end()) {
            // append new state to q table
            std::pair<Observation, vector<double>> mypair;
            mypair = make_pair(observation, q_table_row);
            q_table.insert(mypair);
        }
    }

    int Agent::choose_action(const Observation& observation) {
        check_state_exist(observation);
        // action selection
        srand((unsigned)time(NULL));
        double random = rand() / double(RAND_MAX);
        int action;
        if(random < epsilon){
            // choose best action
            auto it = q_table.find(observation);
            vector<double> tmp = it -> second;
            vector<int> index_of_maxvalue;
            index(tmp, index_of_maxvalue);
            action = index_of_maxvalue[ rand() % index_of_maxvalue.size() ];
        }
        else {
            // choose random action
            action = rand() % nActions;
        }
        return action;
    }

    // 找到每一行内所有相同最大值的下标，放到index_of_maxvalue，之后从index_of_maxvalue中随机选择一个index
    void Agent::index(const vector<double>& myvec, vector<int>& index_of_maxvalue) {
        auto biggest = max_element(std::begin(myvec), std::end(myvec));
        for(int i=0; i<myvec.size(); i++) {
            if(myvec[i]==(*biggest)) {
                index_of_maxvalue.push_back(i);
            }
        }
    }

    void Agent::learn(const Observation& s, const int& a, const double& r, const Observation& s_, const int& a_, const bool& done) {
    check_state_exist(s_);
    double q_predict = (q_table.find(s) -> second)[a];
    // auto it = q_table.find(s);
    // vector<double> tmp = it -> second;
    // double q_predict = tmp[a];
    double q_target;
    if (!done) {
        //q_target = r + gamma * q_table[s_][a_];  // next state is not terminal
        q_target = r + gamma * (q_table.find(s_) -> second)[a_];  // next state is not terminal
    }
    else {
        q_target = r;  // next state is terminal
    }
    q_predict += lr * (q_target - q_predict);  // update
    (q_table.find(s) -> second)[a] = q_predict;
} */

}