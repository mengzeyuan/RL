#include "env.hpp"

namespace nlsr {

Env::Env(ConfParameter& ConfParameter)
: conf_parameter(ConfParameter)
//, m_scheduler(scheduler)
{
    reset();
}

void Env::reset()
{
    this->conf_parameter.setInfoInterestInterval(defaultHelloInterval);
    this->conf_parameter.setRoutingCalcInterval(defaultRoutingCalcInterval);
    this->is_done = false;
    this->episodes++;
    this->current_episode_frames = 0;
    this->total_reward = 0;
}

void Env::step(const int &action)
{
    this->current_episode_frames++;
    uint32_t curHelloInterval = conf_parameter.getInfoInterestInterval();
    uint32_t curRoutingCalcInterval = conf_parameter.getRoutingCalcInterval();
    switch (action)
        {
        case 0: //increase HelloInterval
            if(curHelloInterval<88){
                this->conf_parameter.setInfoInterestInterval(curHelloInterval+2);
            }
        break;
        case 1: //decrease HelloInterval
            if(curHelloInterval>5){
                this->conf_parameter.setInfoInterestInterval(curHelloInterval-2);
            }
        break;
        case 2: //increase RoutingCalculateInterval
            if(curRoutingCalcInterval<20){
                this->conf_parameter.setRoutingCalcInterval(curRoutingCalcInterval+2);
            }
        break;
        case 3: //decrease RoutingCalculateInterval
            if(curRoutingCalcInterval>5){
                this->conf_parameter.setRoutingCalcInterval(curRoutingCalcInterval-2);
            }
        break;

        default:
            break;
        }
}

double Env::get_reward()
{
    double reward;
    this->is_done = false;
    interestHitRatio = (double)ns3::ndn::Consumer::numInData/ns3::ndn::Consumer::numOutInterests;
    interestTimeoutRatio = (double)ns3::ndn::Consumer::numTimeOutInterests/ns3::ndn::Consumer::numOutInterests;
    //std::cout<< "进入Env::getRewardAndObservation" <<std::endl;
    if(interestHitRatio > 0.9 && interestTimeoutRatio < 0.2) {
        reward = 5;
        this->is_done = true;
        reset();
    }
    else {
        double distance = abs(0.9-interestHitRatio) + abs(interestTimeoutRatio-0.2);
        reward = -distance;
    }
    this->total_reward += reward;
    return this->total_reward;
}

vector<double> Env::return_state() {
    interestHitRatio = (double)ns3::ndn::Consumer::numInData/ns3::ndn::Consumer::numOutInterests;
    interestTimeoutRatio = (double)ns3::ndn::Consumer::numTimeOutInterests/ns3::ndn::Consumer::numOutInterests;
    vector<double> state;
    state.push_back(interestHitRatio);
    state.push_back(interestTimeoutRatio);
    return state;
}

bool Env::get_is_done() {
    return this->is_done;
}

}