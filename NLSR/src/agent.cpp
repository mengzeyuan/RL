#include "agent.hpp"
#include <algorithm>


namespace nlsr{

    Agent::Agent(){

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
}