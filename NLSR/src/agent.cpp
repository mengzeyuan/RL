#include "agent.hpp"

namespace nlsr{

    //求最大值下标
    int Agent::argmax (vector<double> array) {
        int index = 0;
        double max_value = array[0];
        for (unsigned int i = 1; i < array.size(); ++i) {
            if (array[i] > max_value) {
                max_value = array[i];
                index = i;
            }
        }
        return index; 
    }
    int Agent::choose_action (vector<double> input) {
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
	        last_prediction = vector<double>({-1, -1, -1, -1});
        } else {
	        last_prediction = this->net.predict(input);
            action = argmax(last_prediction);
        }
        return action;
    }
    void Agent::store_mem (vector<double> current_state, int action, int reward, vector<double> next_state, bool is_done) {
        mem.store(current_state, action, reward, next_state, is_done);
    }
    double Agent::max (vector<double> array) {
        double max_val = array[0];
        for (unsigned int i = 1; i < array.size(); ++i) {
            if (array[i] > max_val) {
                max_val = array[i];
            }
        }
        return max_val;
    }
    void Agent::train () {
        // sample minibatch
        for (unsigned int i = 0; i < batches; ++i) {
            mem.random(); 
            vector<double> current_state = mem.current_state;
            int action = mem.action;
            int reward = mem.reward;
            vector<double> next_state = mem.next_state; 
            bool is_done = mem.is_done;
            // train
            double y;
            if (is_done) {
                y = reward;
            } else {
                y = reward + (0.99 * max(this->target_net.predict(next_state)));
            }
            vector<double> target = this->net.predict(current_state);
            target[action] = y; 
            this->net.backprop(current_state, target, true);
        }
        if (frames % this->targetFreqUpdate == 0) {
            //可以用export_net和read_net代替
            this->target_net = this->net; 
        }
    }

    void Agent::initialize_mem(const string& router_name) {
        mem.initialize(batches, router_name);
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