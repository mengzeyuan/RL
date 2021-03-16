#ifndef RL_HPP
#define RL_HPP

#include "agent.hpp"
#include "env.hpp"

#include <ndn-cxx/util/scheduler.hpp>

namespace nlsr {

class RL {
public:
    RL(ConfParameter& ConfParameter, int BATCHES = 32, double LR = 0.0005, int MEM_CAP = 1000000, int FRAME_REACH = 1000, int TARGET_UPDATE = 500, vector<int> layout={2, 150, 150, 4})
    :m_env(ConfParameter), m_agent(layout, LR, MEM_CAP, FRAME_REACH, TARGET_UPDATE, BATCHES)
    {
    }
    void startRL(double seconds, const ndn::Name& routerName); //设置开始时间
    void update_1();
    void update_2(const vector<double>& current_state);
    void update_3(const int& action, const vector<double>& current_state);
    /* void update_2(const Observation& observation, const int& action);
    void update_3(const Observation& observation, const int& action); */

private:
    Env m_env;
    Agent m_agent;
};

}

#endif