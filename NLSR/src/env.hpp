#ifndef ENV_HPP
#define ENV_HPP

#include "conf-parameter.hpp"
#include "../../apps/ndn-consumer.hpp"

#include <ndn-cxx/util/scheduler.hpp>
#include <vector>
using namespace std;

namespace nlsr{

#define defaultHelloInterval 10
#define defaultRoutingCalcInterval 5

class Env
{
public:
    Env(ConfParameter& ConfParameter);
    void reset();
    void step(const int &action);
    double get_reward();
    vector<double> return_state();
    bool get_is_done();
    int episodes = 0;
    int current_episode_frames;
    double total_reward;

private:
    ConfParameter conf_parameter;
    //ndn::Scheduler& m_scheduler;
    double interestHitRatio, interestTimeoutRatio;  //state
    bool is_done;
};
    
}

#endif