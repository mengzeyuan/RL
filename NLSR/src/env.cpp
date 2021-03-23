#include "env.hpp"
using namespace std;

namespace nlsr {

Env::Env(ConfParameter& ConfParameter)
: conf_parameter(ConfParameter)
{
    reset();
}

void Env::reset()
{
    this->conf_parameter.setInfoInterestInterval(defaultHelloInterval);
    //this->conf_parameter.setRoutingCalcInterval(defaultRoutingCalcInterval);
    this->is_done = false;
    this->episodes++;
    this->current_episode_frames = 0;
    this->total_reward = 0;
}

void Env::step(const int &action)
{
    this->current_episode_frames++;
    current_hello_interval = ConfParameter::rl_hello_interval;
    //uint32_t curRoutingCalcInterval = conf_parameter.getRoutingCalcInterval();
    switch (action)
    {
    case 0: //increase HelloInterval
        if(current_hello_interval<88){
            current_hello_interval+=2;
            this->conf_parameter.setInfoInterestInterval((uint32_t)current_hello_interval);
        }
    break;
    case 1: //decrease HelloInterval
        if(current_hello_interval>5){
            current_hello_interval-=2;
            this->conf_parameter.setInfoInterestInterval((uint32_t)current_hello_interval);
        }
    break;
    /* case 2: //increase RoutingCalculateInterval
        if(curRoutingCalcInterval<20){
            this->conf_parameter.setRoutingCalcInterval(curRoutingCalcInterval+2);
        }
    break;
    case 3: //decrease RoutingCalculateInterval
        if(curRoutingCalcInterval>5){
            this->conf_parameter.setRoutingCalcInterval(curRoutingCalcInterval-2);
        }
    break; */

    default:
        break;
    }
}

double Env::get_reward()
{
    double reward;
    this->is_done = false;
    double rtt = ns3::ndn::Consumer::current_rtt;
    static double min_rtt = rtt;
    if(min_rtt>rtt) {
        min_rtt = rtt;
    }
    double interestHitRatio = (double)ns3::ndn::Consumer::num_in_data/ns3::ndn::Consumer::num_out_interests;
    double rtt_ratio = rtt/min_rtt;
    if(interestHitRatio>0.8 && current_episode_frames>30) {
        reward = 20;
        this->is_done = true;
        this->total_reward += reward;
        //cout<<"Episodes: "<<episodes<<" Current episode frames: "<<current_episode_frames<<" Reward:"<<total_reward<<endl;
        reset();
    }
    else {
        reward = (interestHitRatio-0.8)*3 - rtt_ratio*5;
        this->total_reward += reward;
    }
    //根据上述值决定reward
    /* if(interestHitRatio > 0.7 && interestTimeoutRatio < 0.2) {
        reward = 5;
        this->is_done = true;
        reset();
    }
    else {
        double distance = abs(0.9-interestHitRatio) + abs(interestTimeoutRatio-0.2);
        reward = -distance;
    } */

    return reward;
}

vector<double> Env::return_state() {
    num_timeout_hello = (double)HelloProtocol::num_timeout_hello;
    num_timeout_interest = (double)ns3::ndn::Consumer::num_time_out_interests;
    current_rtt = ns3::ndn::Consumer::current_rtt;
    vector<double> state{num_timeout_hello, num_timeout_interest, current_hello_interval, current_rtt};
    return state;
}

bool Env::get_is_done() {
    return this->is_done;
}

}